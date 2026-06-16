Run the full TIRX test suite.

## Steps

1. Select the least busy GPU to avoid conflicts:
   ```bash
   export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
   ```

2. Start the GPU monitor in the background so we can detect if anyone else lands on the same GPU mid-run:
   ```bash
   GPU_LOG="/tmp/tir_test_gpu_${CUDA_VISIBLE_DEVICES}.log"
   bash .agents/scripts/monitor_gpu.sh --gpu "$CUDA_VISIBLE_DEVICES" --interval 5 --log "$GPU_LOG" &
   MON_PID=$!
   trap 'kill $MON_PID 2>/dev/null' EXIT
   ```

3. Run the full test suite with xdist parallelism:
   ```bash
   pytest tests/python/tirx/ -n auto
   ```

4. Stop the monitor and check for foreign GPU usage during the run:
   ```bash
   kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
   grep -E 'FOREIGN USER|\[FOREIGN\]' "$GPU_LOG" || echo "no foreign GPU usage observed"
   ```

5. Report results: total passed, failed, skipped, errors. If any foreign-user events are present in step 4, mention them — flaky failures should be re-evaluated on a clean GPU before being attributed to code changes.

## Failure triage rules

**CRITICAL: Never pipe test output to `tail` or `grep` when diagnosing failures. Always capture and read full logs.**

Classify every failure into one of these categories:

- **A — Environment/import error**: Module not found, missing dependency, collection error. These are not caused by code changes.
- **B — Real kernel correctness regression**: Assertion failures (cosine_sim, numerical diff), `CUDA: unspecified launch failure`, or wrong results. **These MUST be investigated and fixed if caused by current changes.**
- **C — Secondary xdist crash**: `KeyError: <WorkerController gwXX>` after a worker abort. The KeyError itself is noise — find the underlying cause (usually category B in another worker).

**Never dismiss a failure as "pre-existing" without evidence.** If a test fails:
1. Check whether the test touches code you changed.
2. If unclear, verify on the parent commit before claiming pre-existing.
3. All failures caused by current changes MUST be fixed — not deferred.
