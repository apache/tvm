Run the full TIRX test suite.

## Steps

Run all commands below in the same Bash shell from the TVM repository root.

1. Point Python at this workspace's repos and select every eligible idle Blackwell GPU.
   `TIR_TEST_GPUS` may explicitly provide physical GPU IDs, and
   `TIR_TEST_NUM_GPUS` may optionally cap the automatically selected count. Otherwise,
   selection uses all eligible GPUs, ordered by least memory in use.
   ```bash
   export WORKSPACE=/path/to/workspace
   export TIR_TEST_NUM_GPUS="${TIR_TEST_NUM_GPUS:-}"
   export TIR_TEST_MIN_FREE_GB="${TIR_TEST_MIN_FREE_GB:-32}"
   export TIR_TEST_MAX_USED_MIB="${TIR_TEST_MAX_USED_MIB:-256}"
   export TIR_TEST_XDIST_WORKERS="${TIR_TEST_XDIST_WORKERS:-16}"

   [[ -z "$TIR_TEST_NUM_GPUS" || "$TIR_TEST_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || {
     echo "TIR_TEST_NUM_GPUS must be empty or a positive integer" >&2
     exit 2
   }
   [[ "$TIR_TEST_MIN_FREE_GB" =~ ^[0-9]+$ ]] || {
     echo "TIR_TEST_MIN_FREE_GB must be a non-negative integer" >&2
     exit 2
   }
   [[ "$TIR_TEST_MAX_USED_MIB" =~ ^[0-9]+$ ]] || {
     echo "TIR_TEST_MAX_USED_MIB must be a non-negative integer" >&2
     exit 2
   }
   [[ "$TIR_TEST_XDIST_WORKERS" =~ ^[1-9][0-9]*$ ]] || {
     echo "TIR_TEST_XDIST_WORKERS must be a positive integer" >&2
     exit 2
   }

   if [[ -n "${TIR_TEST_GPUS:-}" ]]; then
     IFS=, read -r -a TIR_TEST_SELECTED_GPUS <<< "${TIR_TEST_GPUS// /}"
   else
     mapfile -t TIR_TEST_SELECTED_GPUS < <(
       nvidia-smi \
         --query-gpu=index,memory.total,memory.used,utilization.gpu,compute_cap \
         --format=csv,noheader,nounits \
       | awk -F, \
           -v min_free_mib="$((TIR_TEST_MIN_FREE_GB * 1024))" \
           -v max_used_mib="$TIR_TEST_MAX_USED_MIB" '
           {
             for (i = 1; i <= 5; ++i) {
               gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
             }
             free_mib = $2 - $3
             if (($5 + 0) >= 10 && free_mib >= min_free_mib &&
                 ($3 + 0) <= max_used_mib && ($4 + 0) == 0) {
               print $1, $3
             }
           }' \
       | sort -k2,2n -k1,1n \
       | awk '{print $1}'
     )
     if [[ -n "$TIR_TEST_NUM_GPUS" ]] && \
        (( ${#TIR_TEST_SELECTED_GPUS[@]} > TIR_TEST_NUM_GPUS )); then
       TIR_TEST_SELECTED_GPUS=(
         "${TIR_TEST_SELECTED_GPUS[@]:0:TIR_TEST_NUM_GPUS}"
       )
     fi
   fi

   (( ${#TIR_TEST_SELECTED_GPUS[@]} > 0 )) || {
     echo "no eligible CUDA compute capability >= 10 GPU is available" >&2
     exit 1
   }
   for gpu in "${TIR_TEST_SELECTED_GPUS[@]}"; do
     [[ "$gpu" =~ ^[0-9]+$ ]] || {
       echo "invalid physical GPU ID: $gpu" >&2
       exit 2
     }
   done
   if [[ -z "${TIR_TEST_GPUS:-}" && -n "$TIR_TEST_NUM_GPUS" ]] && \
      (( ${#TIR_TEST_SELECTED_GPUS[@]} < TIR_TEST_NUM_GPUS )); then
     echo "warning: requested $TIR_TEST_NUM_GPUS GPUs, using ${#TIR_TEST_SELECTED_GPUS[@]}"
   fi

   export CUDA_VISIBLE_DEVICES
   CUDA_VISIBLE_DEVICES="$(IFS=,; printf '%s' "${TIR_TEST_SELECTED_GPUS[*]}")"
   export TIRX_KERNEL_TEST_MIN_FREE_GB="$TIR_TEST_MIN_FREE_GB"
   export PYTHONPATH="${WORKSPACE}/tirx-kernels:${WORKSPACE}/tvm/python"
   export TVM_LIBRARY_PATH="${WORKSPACE}/tvm/build/lib"

   PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
   [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
     echo "set PYTHON_BIN to the test environment's Python executable" >&2
     exit 1
   }
   export PATH="$(dirname "$PYTHON_BIN"):$PATH"
   "$PYTHON_BIN" -c 'import pytest, torch, tvm' >/dev/null || {
     echo "PYTHON_BIN must provide pytest and torch and import this TVM worktree" >&2
     exit 1
   }
   if ! command -v ninja >/dev/null; then
     NINJA_BIN_DIR="$("$PYTHON_BIN" -c 'import ninja; print(ninja.BIN_DIR)' 2>/dev/null || true)"
     if [[ -n "$NINJA_BIN_DIR" && -x "$NINJA_BIN_DIR/ninja" ]]; then
       export PATH="$NINJA_BIN_DIR:$PATH"
     fi
   fi
   command -v ninja >/dev/null || {
     echo "ninja is required by kernel JIT setup but is not on PATH" >&2
     exit 1
   }

   echo "selected physical GPUs: $CUDA_VISIBLE_DEVICES"
   for logical_id in "${!TIR_TEST_SELECTED_GPUS[@]}"; do
     echo "  cuda:$logical_id -> physical GPU ${TIR_TEST_SELECTED_GPUS[$logical_id]}"
   done
   ```

   Automatic selection uses every eligible idle device. Set
   `TIR_TEST_NUM_GPUS=4` to cap the count or `TIR_TEST_GPUS=3,4,5,6` to override
   selection with exact physical IDs.

2. Start one GPU monitor per selected physical GPU. A baseline foreign process
   is a selection race; stop and select a different GPU before running tests.
   ```bash
   TIR_TEST_MONITOR_PIDS=()
   TIR_TEST_GPU_LOGS=()

   cleanup_tir_test_monitors() {
     local pid
     for pid in "${TIR_TEST_MONITOR_PIDS[@]}"; do
       kill "$pid" 2>/dev/null || true
     done
     for pid in "${TIR_TEST_MONITOR_PIDS[@]}"; do
       wait "$pid" 2>/dev/null || true
     done
   }
   trap cleanup_tir_test_monitors EXIT

   for gpu in "${TIR_TEST_SELECTED_GPUS[@]}"; do
     log="/tmp/tir_test_gpu_${gpu}_$$.log"
     bash .agents/scripts/monitor_gpu.sh \
       --gpu "$gpu" --interval 5 --log "$log" &
     TIR_TEST_MONITOR_PIDS+=("$!")
     TIR_TEST_GPU_LOGS+=("$log")
   done
   sleep 1

   baseline_conflict=0
   for log in "${TIR_TEST_GPU_LOGS[@]}"; do
     if grep -q '\[FOREIGN\]' "$log"; then
       echo "foreign process present at baseline: $log" >&2
       baseline_conflict=1
     fi
   done
   if (( baseline_conflict )); then
     cleanup_tir_test_monitors
     trap - EXIT
     exit 1
   fi
   ```

3. Import gate: bench workloads. Fail fast if any kernel listed in
   `workloads.yaml` fails to import.
   ```bash
   "$PYTHON_BIN" -m tirx_kernels.bench_suite --check-imports
   ```
   A non-zero exit means a pinned workload kernel failed to import. Fix it before
   proceeding.

4. Full kernel import gate for correctness-suite coverage:
   ```bash
   "$PYTHON_BIN" -m tirx_kernels.registry --cc 10 --strict
   ```

5. Run the full test suite with xdist parallelism:
   ```bash
   "$PYTHON_BIN" -m pytest tests/python/tirx/ -n "$TIR_TEST_XDIST_WORKERS"
   ```

   `tests/python/tirx/conftest.py` assigns each xdist worker's PyTorch current
   device round-robin across the visible GPUs. Registry correctness tests also
   take a per-device lock, so at most one large registered kernel runs on each
   GPU concurrently. Tests that explicitly use `tvm.cuda(0)` still run on the
   first selected physical GPU. MegaMoE remains skipped because it requires its
   dedicated multi-process scheduler, not ordinary xdist device assignment.

6. Stop every monitor and check every selected GPU for foreign usage:
   ```bash
   cleanup_tir_test_monitors
   trap - EXIT

   foreign_usage=0
   for i in "${!TIR_TEST_GPU_LOGS[@]}"; do
     gpu="${TIR_TEST_SELECTED_GPUS[$i]}"
     log="${TIR_TEST_GPU_LOGS[$i]}"
     if grep -qE 'FOREIGN USER|\[FOREIGN\]' "$log"; then
       echo "GPU $gpu foreign-process events:"
       grep -E 'FOREIGN USER|\[FOREIGN\]' "$log"
       foreign_usage=1
     else
       echo "GPU $gpu: no foreign GPU usage observed"
     fi
   done
   ```

7. Report results: selected physical GPUs, elapsed time, total passed, failed,
   skipped, and errors, plus both import-gate results. If any foreign-user events
   are present in step 6, mention the affected physical GPUs. Flaky failures
   should be re-evaluated on clean GPUs before being attributed to code changes.

## Failure triage rules

**CRITICAL: Never pipe test output to `tail` or `grep` when diagnosing failures.
Always capture and read full logs.**

Classify every failure into one of these categories:

- **A - Environment/import error**: Module not found, missing dependency,
  collection error. These are not caused by code changes.
- **B - Real kernel correctness regression**: Assertion failures (`cosine_sim`,
  numerical diff), `CUDA: unspecified launch failure`, or wrong results. These
  MUST be investigated and fixed if caused by current changes.
- **C - Secondary xdist crash**: `KeyError: <WorkerController gwXX>` after a
  worker abort. The KeyError itself is noise; find the underlying cause (usually
  category B in another worker).

Never dismiss a failure as pre-existing without evidence. If a test fails:

1. Check whether the test touches code you changed.
2. If unclear, verify on the parent commit before claiming it is pre-existing.
3. Fix every failure caused by current changes; do not defer it.
