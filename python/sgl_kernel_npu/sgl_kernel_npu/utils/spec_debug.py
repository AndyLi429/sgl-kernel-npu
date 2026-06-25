"""Env-gated diagnostics for the speculative-decoding state-rollback path on
hybrid (GDN/Mamba) models.

Default OFF. When disabled, every entry point is a single env/bool check that
returns immediately -- no device sync, no host copy, no numerical change. There
is NO torch import in this module (the tiny per-request tensors are read via
``.tolist()`` on whatever is passed in), so it is safe to import anywhere.

Enable on a debug / canary instance only::

    export SGLANG_NPU_SPEC_DEBUG=1            # turn on  (MUST be exported!)
    export SGLANG_NPU_SPEC_DEBUG_EVERY=200    # (optional) periodic-summary period

Key lines are written to BOTH the logger AND stderr (flushed), so they survive
any logging-config filtering as long as the process stderr is captured
(``... > sglang.log 2>&1``).

What it catches
---------------
1. NEGATIVE ROLLBACK OFFSET ("failure surface A", the prime suspect):
   the conv kernels compute ``conv_state_token_offset = num_accepted_tokens - 1``
   (causal_conv1d.py). If ``num_accepted_tokens`` ever reaches the kernel as 0,
   the offset is -1 and the kernel reads the conv slot BEFORE this request's
   region -> a cross-request recurrent-state read -> corrupted state -> loop.
2. LOW-ACCEPT TAIL (validates the reproduction direction): periodic accept
   histogram; the bug needs the low tail, which accept==1 workloads never make.
3. CONV vs SSM ROLLBACK INDEX (the deeper desync suspect): logs both rollback
   index tensors with slot ids so they can be diffed offline.

NOTE: enabling adds a per-call device->host sync (perturbs timing). It reliably
catches the DATA bugs above, but may mask pure TIMING races (overlap / spec-v2);
for those use ASCEND_LAUNCH_BLOCKING=1 and SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0.
"""

import logging
import os
import sys
import threading

logger = logging.getLogger("sgl_kernel_npu.spec_debug")

_enabled = None
_every = None
_count = 0
_name_counts = {}
_lock = threading.Lock()
_announced = False


def is_enabled():
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get("SGLANG_NPU_SPEC_DEBUG", "0").lower() not in (
            "0",
            "",
            "false",
            "no",
        )
    return _enabled


def _period():
    global _every
    if _every is None:
        try:
            _every = max(1, int(os.environ.get("SGLANG_NPU_SPEC_DEBUG_EVERY", "200")))
        except ValueError:
            _every = 200
    return _every


def _emit(msg, warn=False):
    # stderr is captured into sglang.log via 2>&1 even if the named logger is
    # filtered by sglang's logging setup -> guaranteed visibility.
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass
    if warn:
        logger.warning(msg)
    else:
        logger.info(msg)


def _announce_once():
    global _announced
    if not _announced:
        _announced = True
        _emit(
            "[spec_debug] ENABLED via SGLANG_NPU_SPEC_DEBUG; per-call device->host "
            "sync added -- use on a debug/canary instance only.",
            warn=True,
        )


def record_accept(num_accepted_tokens, state_indices=None, tag="conv"):
    """num_accepted_tokens / state_indices: (batch,) int tensors. See module docstring."""
    global _count
    if not is_enabled() or num_accepted_tokens is None:
        return
    try:
        _announce_once()
        with _lock:
            _count += 1
            c = _count

        na = num_accepted_tokens.detach().reshape(-1).tolist()  # tiny (<=bs); one sync
        bs = len(na)
        na_min = min(na) if na else 1

        if na_min < 1:
            slots = None
            if state_indices is not None:
                try:
                    si = state_indices.detach().reshape(-1).tolist()
                    slots = [si[i] for i, v in enumerate(na) if v < 1 and i < len(si)]
                except Exception:
                    slots = None
            n_bad = sum(1 for v in na if v < 1)
            _emit(
                f"[spec_debug:{tag}] NEGATIVE OFFSET min(num_accepted)={na_min} -> "
                f"offset={na_min - 1} bs={bs} n_bad={n_bad} bad_slots={slots} accept={na}",
                warn=True,
            )
        elif c % _period() == 0:
            hist = {}
            for v in na:
                hist[v] = hist.get(v, 0) + 1
            _emit(
                f"[spec_debug:{tag}] call={c} bs={bs} "
                f"accept_count_hist={dict(sorted(hist.items()))}"
            )
    except Exception as e:  # instrumentation must never break the run
        logger.debug("[spec_debug] record_accept skipped: %s", e)


def record_rollback(name, step_idx_tensor, slot_tensor=None, draft_token_num=None):
    """Log a rollback index tensor (+ slot ids) for offline conv-vs-ssm desync diff."""
    if not is_enabled() or step_idx_tensor is None:
        return
    try:
        _announce_once()
        with _lock:
            n = _name_counts.get(name, 0) + 1
            _name_counts[name] = n
        if n % _period() != 0:
            return
        steps = step_idx_tensor.detach().reshape(-1).tolist()
        slots = None
        if slot_tensor is not None:
            try:
                slots = slot_tensor.detach().reshape(-1).tolist()
            except Exception:
                slots = None
        if steps:
            n_neg = sum(1 for v in steps if v < 0)
            _emit(
                f"[spec_debug:{name}] tick={n} draft={draft_token_num} "
                f"step_min={min(steps)} step_max={max(steps)} n_neg={n_neg} "
                f"step_idx={steps} slots={slots}"
            )
    except Exception as e:
        logger.debug("[spec_debug] record_rollback skipped: %s", e)
