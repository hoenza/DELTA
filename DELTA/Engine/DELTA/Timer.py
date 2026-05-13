import torch
from collections import deque, defaultdict
import os, json, time
from pathlib import Path


def _get_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    except Exception:
        return 0


class Timer:
    def __init__(self, timing_enabled=False, timing_model_forward_enabled=False, timing_log_dir='/tmp/op_times', ):
        # self._timing_enabled = False  # flip to False to disable per-call timing
        self._timing_enabled = timing_enabled
        # self._timing_model_forward_enabled = False
        self._timing_model_forward_enabled = timing_model_forward_enabled
        self._op_stream = deque(maxlen=1000000)      # ordered stream of (ts_ms, op, ms)
        self._op_idx = 0 
        
        self._timing_autoflush_enabled    = True
        self._timing_autoflush_ratio      = 0.90   # flush when len >= 90% of maxlen
        self._timing_autoflush_min        = 512    # also require at least this many samples
        self._timing_autoflush_cooldown_s = 5.0    # don't flush the same op more often than this
        # self._timing_log_dir              = "/tmp/op_times"
        self._timing_log_dir = timing_log_dir
        self._stream_autoflush_min = 256        # flush the stream when it accumulates this many rows
        self._last_stream_flush = 0.0

        # last-flush timestamps per op
        self._timing_last_flush = defaultdict(lambda: 0.0)
        Path(self._timing_log_dir).mkdir(parents=True, exist_ok=True)
    
    def get_timing_model_forward_enabled(self):
        return self._timing_model_forward_enabled
    
    def get_timing_enabled(self):
        return self._timing_enabled
    
    def record_cuda_ms(self, name: str, start_evt: torch.cuda.Event, end_evt: torch.cuda.Event):
        # end_evt must be synchronized by caller
        try:
            ms = start_evt.elapsed_time(end_evt)
        except Exception:
            ms = float("nan")
        # self._op_timings[name].append(ms)               # existing behavior (per-op stats)
        ts_ms = int(time.time() * 1000)                 # capture event time NOW
        self._op_stream.append((ts_ms, name, ms))
        # print(self._op_stream)
    
    def get_timings(self, name: str):
        """Return a copy of the raw per-call timings (ms) for an op."""
        return list(self._op_timings.get(name, []))

    def clear_timings(self, name: str = None):
        """Clear one op's timings or all."""
        if name is None:
            for k in self._op_timings: self._op_timings[k].clear()
        else:
            self._op_timings.get(name, deque()).clear()
    
    def _save_one_op_csv(self, op_name: str, reset: bool = True) -> int:
        # Single file per torchrun process; name includes LOCAL_RANK (fallback RANK -> 0)
        rank = _get_local_rank()
        path = os.path.join(self._timing_log_dir, f"op_times_rank{rank}.csv")
        # print(f'path: {path}')
        is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        written = 0
        with open(path, "a") as f:
            if is_new:
                f.write("ts_ms,idx,op,ms\n")
            # IMPORTANT: write EVERYTHING currently in the ordered stream to preserve global order
            while self._op_stream:
                ts_ms, name, ms = self._op_stream.popleft()
                f.write(f"{ts_ms},{self._op_idx},{name},{ms}\n")
                self._op_idx += 1
                written += 1
        if reset:
            # keep compatibility with existing autoflush (clear that op’s per-op deque only)
            dq = self._op_timings.get(op_name)
            if dq:
                dq.clear()
        return written

    def _save_one_op_jsonl(self, op_name: str, reset: bool = True, extra_meta: dict | None = None) -> int:
        dq = self._op_timings.get(op_name)
        if not dq:
            return 0
        samples = list(dq)
        if not samples:
            return 0
        extra_meta = extra_meta or {}
        path = os.path.join(self._timing_log_dir, f"{op_name}.jsonl")
        now_ms = int(time.time() * 1000)
        with open(path, "a") as f:
            for i, ms in enumerate(samples):
                rec = {"ts_ms": now_ms, "idx": i, "ms": float(ms), "op": op_name}
                rec.update(extra_meta)
                f.write(json.dumps(rec) + "\n")
        if reset:
            dq.clear()
        return len(samples)
    
    def maybe_autoflush_all_timings(self):
        if not self._timing_autoflush_enabled:
            return
        now = time.time()
        if len(self._op_stream) >= self._stream_autoflush_min:
            # print('if1')
            if (now - self._last_stream_flush) >= self._timing_autoflush_cooldown_s:
                # print('if2')
                written = self._save_one_op_csv(op_name="__stream__", reset=False)
                if written > 0:
                    self._last_stream_flush = now
