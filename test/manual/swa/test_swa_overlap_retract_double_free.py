"""Repro for SWA hybrid-attention KV double-free assertion under retract+overlap.

Symptom (Gemma3/Gemma4-class hybrid SWA models):

    AssertionError at python/sglang/srt/mem_cache/allocator/swa.py
    in SWATokenToKVPoolAllocator.free():
        assert (
            self.full_attn_allocator.available_size()
            <= self.full_attn_allocator.size
        )

`available_size > size` can only happen if the same KV index is freed
twice — each duplicate free re-adds the index to the free list and
inflates the count past the pool total. The note in `free()` says the
API is intentionally non-idempotent, so a second free is not silently
absorbed but trips the assert and crashes the scheduler.

Trigger conditions (all required, this test reproduces them):

  1. Hybrid-attention model that uses `SWATokenToKVPoolAllocator`
     (separate full + swa sub-allocators with `full_to_swa_index_mapping`).
  2. Overlap scheduling (default) — `process_batch_result_decode` defers
     freeing into a `free_group` and relies on
     `req.is_retracted`/`req.finished()` to skip the per-req decode-side
     free. See `scheduler_components/batch_result_processor.py:632-681`.
  3. Retract under KV pressure — `release_req` in `schedule_batch.py:1547`
     calls `release_kv_cache` which already frees the request's full-pool
     range (via `swa_radix_cache.cache_finished_req` /
     `tree_cache.token_to_kv_pool_allocator.free`). The overlap guard
     (`req.is_retracted`) is supposed to suppress the second free in the
     decode-side `free_group`, but on the SWA double-pool full side that
     coverage is incomplete and a second `free()` of the same indices
     reaches the allocator.

The swa side is implicitly idempotent because `free_swa()` filters with
`swa_indices[swa_indices > 0]` after re-zeroing the mapping
(`allocator/swa.py:345-357`). The full side has no such guard, so it is
the one that asserts.

This test is in `test/manual/` because the underlying bug is unfixed on
main as of writing — it reproduces the crash, it does not pass yet. Move
to `test/registered/` once the state-machine fix lands.

How to run:

    python3 test/manual/swa/test_swa_overlap_retract_double_free.py

Expected (current, pre-fix): the scheduler subprocess dies with
`AssertionError` at `allocator/swa.py:323`. The test fails by detecting
`self.process.poll() is not None`.

Expected (post-fix): all requests return 200 and the scheduler stays
alive across `_NUM_REQUESTS` waves of forced retract.
"""

import concurrent.futures
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


# Smallest hybrid-SWA model we can comfortably run on a single GPU. Gemma3
# uses the same `SWAKVPool` + `SWATokenToKVPoolAllocator` path that Gemma4
# does in production, so the failing free() lives in identical code.
SWA_MODEL = "google/gemma-3-4b-it"

# Shared long context plus a per-request tail keeps decode batches fat so
# `check_decode_mem` keeps failing and the scheduler keeps re-entering
# `retract_decode`. Each request also generates enough tokens to stay in
# decode for many overlap iterations.
_SHARED_PREFIX = (
    "You are a careful assistant. Context: "
    + ("the quick brown fox jumps over the lazy dog. " * 400)
)
_QUESTION_TAILS = [
    " Q: Summarize the passage in one sentence.\n",
    " Q: List three colors mentioned implicitly.\n",
    " Q: Translate the first sentence to French.\n",
    " Q: Continue the story for two more sentences.\n",
]
_NUM_REQUESTS = 32
_CONCURRENCY = 16
_MAX_NEW_TOKENS = 512


class TestSWAOverlapRetractDoubleFree(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SWA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Forced periodic retract every 3 scheduler iterations — turns a
        # rare workload-dependent crash into a deterministic one. Combined
        # with a tight `mem_fraction_static` and a high
        # `max_running_requests`, every retract call runs against the SWA
        # double pool with overlap-mode async free in flight.
        env = {
            "SGLANG_TEST_RETRACT": "1",
            "SGLANG_TEST_RETRACT_INTERVAL": "3",
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.55",
                "--max-running-requests",
                "32",
                "--context-length",
                "8192",
                "--random-seed",
                "0",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _fire_one(self, idx: int):
        tail = _QUESTION_TAILS[idx % len(_QUESTION_TAILS)]
        prompt = _SHARED_PREFIX + tail + f" (seed={idx})"
        try:
            r = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": _MAX_NEW_TOKENS,
                    },
                },
                timeout=600,
            )
            return r.status_code == 200, ""
        except Exception as e:
            return False, repr(e)

    def test_no_double_free_assert_under_forced_retract(self):
        """Scheduler must survive forced retract on a hybrid-SWA model.

        The crash is a `SWATokenToKVPoolAllocator.free` assert that takes
        down the scheduler subprocess. We detect it by `process.poll()`
        returning a non-None exit code. Per-request 200 is a stronger
        post-fix bar — under retract pressure some requests can be aborted
        legitimately, so we only treat process death as the hard failure.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=_CONCURRENCY
        ) as ex:
            futs = [ex.submit(self._fire_one, i) for i in range(_NUM_REQUESTS)]
            n_ok = n_fail = 0
            first_fail = ""
            for f in concurrent.futures.as_completed(futs):
                ok, msg = f.result()
                if ok:
                    n_ok += 1
                else:
                    if n_fail == 0:
                        first_fail = msg
                    n_fail += 1

        print(
            f"n_ok={n_ok} n_fail={n_fail} "
            f"server_alive={self.process.poll() is None} "
            f"first_fail={first_fail!r}"
        )
        # The only invariant we enforce: the scheduler did not crash.
        self.assertIsNone(
            self.process.poll(),
            "Scheduler subprocess exited — likely the SWA double-free assert "
            "in allocator/swa.py:free(). See file docstring for analysis.",
        )


if __name__ == "__main__":
    unittest.main()
