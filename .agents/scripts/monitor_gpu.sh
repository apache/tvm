#!/usr/bin/env bash
# Watch a single GPU for foreign processes (anyone other than the current
# user) appearing during a long-running test. Intended companion to
# `/tir-test`: leave this running in a side terminal while pytest runs, and
# it will alert if someone else lands on the same GPU.
#
# Usage:
#   monitor_gpu.sh                       # uses $CUDA_VISIBLE_DEVICES, defaults to 0
#   monitor_gpu.sh --gpu 3               # watch GPU 3
#   monitor_gpu.sh --gpu 3 --interval 2  # poll every 2 seconds
#   monitor_gpu.sh --log /tmp/gpu.log    # also tee to a log file

# Note: deliberately not `set -u` — bash <5.2 errors on `${#assoc[@]}` when
# the associative array is empty.

GPU=""
INTERVAL=5
LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Watch a single GPU for foreign processes (anyone other than the current
user) appearing during a long-running test. Intended companion to
`/tir-test`: leave this running in a side terminal while pytest runs, and
it will alert if someone else lands on the same GPU.

Usage:
  monitor_gpu.sh                       # uses $CUDA_VISIBLE_DEVICES, defaults to 0
  monitor_gpu.sh --gpu 3               # watch GPU 3
  monitor_gpu.sh --gpu 3 --interval 2  # poll every 2 seconds
  monitor_gpu.sh --log /tmp/gpu.log    # also tee to a log file
EOF
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$GPU" ]]; then
  GPU="${CUDA_VISIBLE_DEVICES:-0}"
fi
# Only the first index if CUDA_VISIBLE_DEVICES is a list.
GPU="${GPU%%,*}"
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
  echo "monitor_gpu: GPU must be an integer index (got '$GPU'); pass --gpu <n>" >&2
  exit 2
fi

ME="$(id -un)"

emit() {
  local line="[$(date +'%H:%M:%S')] $*"
  if [[ -n "$LOG" ]]; then
    printf '%s\n' "$line" | tee -a "$LOG" >&2
  else
    printf '%s\n' "$line" >&2
  fi
}

# Returns "pid|user|mem_mib|process_name" lines for compute apps on $GPU.
snapshot() {
  nvidia-smi --id="$GPU" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null \
  | while IFS=, read -r pid pname mem; do
      pid="${pid// /}"
      [[ -z "$pid" ]] && continue
      local user
      user="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
      [[ -z "$user" ]] && user="?"
      pname="${pname# }"
      mem="${mem# }"
      printf '%s|%s|%s|%s\n' "$pid" "$user" "$mem" "$pname"
    done
}

emit "monitor_gpu started: GPU=$GPU interval=${INTERVAL}s user=$ME"

declare -A KNOWN  # pid -> "user|mem|pname"

# Initial snapshot — record everyone we already see as the baseline.
while IFS='|' read -r pid user mem pname; do
  [[ -z "${pid:-}" ]] && continue
  KNOWN[$pid]="$user|$mem|$pname"
  flag=""
  [[ "$user" != "$ME" ]] && flag=" [FOREIGN]"
  emit "baseline pid=$pid user=$user mem=${mem}MiB cmd=$pname$flag"
done < <(snapshot)

if [[ ${#KNOWN[@]} -eq 0 ]]; then
  emit "baseline: GPU $GPU is idle"
fi

trap 'emit "monitor_gpu stopped"; exit 0' INT TERM

heartbeat_due=$(( $(date +%s) + 60 ))

while :; do
  sleep "$INTERVAL"

  declare -A SEEN=()
  while IFS='|' read -r pid user mem pname; do
    [[ -z "${pid:-}" ]] && continue
    SEEN[$pid]=1
    if [[ -z "${KNOWN[$pid]:-}" ]]; then
      flag=""
      [[ "$user" != "$ME" ]] && flag=" *** FOREIGN USER ***"
      emit "NEW pid=$pid user=$user mem=${mem}MiB cmd=$pname$flag"
      KNOWN[$pid]="$user|$mem|$pname"
    fi
  done < <(snapshot)

  for pid in "${!KNOWN[@]}"; do
    if [[ -z "${SEEN[$pid]:-}" ]]; then
      emit "GONE pid=$pid (was: ${KNOWN[$pid]})"
      unset 'KNOWN[$pid]'
    fi
  done
  unset SEEN

  now=$(date +%s)
  if (( now >= heartbeat_due )); then
    foreign=0
    for v in "${KNOWN[@]}"; do
      u="${v%%|*}"
      [[ "$u" != "$ME" ]] && foreign=$((foreign+1))
    done
    emit "heartbeat: ${#KNOWN[@]} process(es) on GPU $GPU (${foreign} foreign)"
    heartbeat_due=$(( now + 60 ))
  fi
done
