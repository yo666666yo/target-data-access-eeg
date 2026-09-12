#!/usr/bin/env bash
# Launch the schema-v4 run matrix with a bounded, dynamically scheduled GPU pool.
#
#   main       2x2 factorial: alignment x target supervision
#   secondary  the two demoted v3 arms, each now matched to the same source
#              partition and optimizer budget as the main conditions
#
# All conditions in the main block share the normalization mode, so the only
# thing that varies there is the two factorial bits.  Jobs are skipped when
# their exact run log already exists, matched on the full run id rather than on
# a fragment of it: several conditions share a normalization string, and
# matching on that alone once let one condition's log suppress another's.
#
# Scheduling is dynamic (xargs -P) rather than striped, because the PhysionetMI
# folds are an order of magnitude longer than the others and a static split
# would leave most workers idle at the end.

set -u

ROOT=/home/USER/eeg-eval
PY=$ROOT/.venv/bin/python
RUNNER=$ROOT/experiments/paper_runner.py
# Overridable so a second driver can work a different slice of the matrix
# concurrently: the PhysionetMI folds are long enough that waiting for them
# before touching the small datasets would leave nothing to check for hours.
TAG=${V4_TAG:-v4}
LOGROOT=$ROOT/logs/$TAG
NORM=outer_train_scaler
SEEDS=${V4_SEEDS:-"0 1 2"}
NSLOTS=${V4_NSLOTS:-4}
GPU_POOL=${V4_GPUS:-"0 2 4 5"}
EPOCHS=${V4_EPOCHS:-50}
ONLY=${V4_ONLY:-}

mkdir -p "$LOGROOT"

# key|dataset|extra runner args|result dir
DATASETS=(
  "2a|BNCI2014_001|--data-source official-gdf --official-gdf-root $ROOT/.cache/bciciv_2a|$ROOT/results/v4_2a"
  "002|BNCI2014_002||$ROOT/results/v4_002"
  "physionet|PhysionetMI||$ROOT/results/v4_physionet"
)

# condition|extra runner args
CONDITIONS=(
  "SRC|"
  "EA|--euclidean_alignment"
  "SUP|--target_supervised"
  "EASUP|--euclidean_alignment --target_supervised"
  "SEL|--legacy_loso"
  "POOL|"
)

# Longest datasets first, so the tail of the queue is short jobs.
order_key() { case "$1" in physionet) echo 0;; 2a) echo 1;; *) echo 2;; esac; }

QUEUE=$LOGROOT/queue.txt
: > "$QUEUE"
NQ=0
for ds in "${DATASETS[@]}"; do
  IFS='|' read -r key dataset dsargs outdir <<< "$ds"
  if [ -n "$ONLY" ] && ! printf '%s\n' $ONLY | grep -qx "$key"; then
    continue
  fi
  mkdir -p "$outdir"
  for cond in "${CONDITIONS[@]}"; do
    IFS='|' read -r cname cargs <<< "$cond"
    norm=$NORM
    [ "$cname" = "POOL" ] && norm=pooled_all_subject_scaler
    for seed in $SEEDS; do
      rid=$("$PY" - "$dataset" "$seed" "$norm" "$cargs" <<'PYEOF'
import sys
sys.path.insert(0, '/home/USER/eeg-eval')
from experiments.schema import build_run_id, protocol_id_v4
dataset, seed, norm, cond = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
print(build_run_id('eegnet', 'equal', dataset, seed, protocol_id_v4(
    '--euclidean_alignment' in cond, '--target_supervised' in cond,
    norm, '--legacy_loso' in cond)))
PYEOF
)
      if [ -f "$outdir/$rid.json" ]; then
        echo "[skip] $key $cname s$seed"
        continue
      fi
      # '|' rather than a tab: tab is IFS whitespace, so `read` would
      # collapse the empty dataset-args field and shift every later field.
      printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "$(order_key "$key")" "$key" "$cname" "$seed" "$dataset" \
        "$dsargs" "$cargs" "$norm" "$outdir" "$rid" >> "$QUEUE"
      NQ=$((NQ + 1))
    done
  done
done

sort -t'|' -k1,1n -o "$QUEUE" "$QUEUE"
echo "[plan] $NQ job(s) queued across $NSLOTS slot(s) on GPUs: $GPU_POOL"
[ "$NQ" -eq 0 ] && exit 0

cat > "$LOGROOT/run_one.sh" <<'ONEEOF'
#!/usr/bin/env bash
set -u
ROOT=/home/USER/eeg-eval
PY=$ROOT/.venv/bin/python
RUNNER=$ROOT/experiments/paper_runner.py
LOGROOT=$LOGROOT_ENV
IFS='|' read -r _ key cname seed dataset dsargs cargs norm outdir rid <<< "$1"
GPUS=($GPU_POOL_ENV)
# Pick a slot deterministically from the shell PID so concurrent workers spread
# across the pool without needing a shared counter.
gpu=${GPUS[$(( $$ % ${#GPUS[@]} ))]}
out="$LOGROOT/${key}_${cname}_s${seed}.log"
echo "[$(date +%H:%M:%S) gpu$gpu] start $key $cname s$seed"
CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$PY" "$RUNNER" --variant eegnet --weighting equal \
    --dataset "$dataset" $dsargs $cargs \
    --normalization "$norm" --seed "$seed" --epochs ${V4_EPOCHS:-50} \
    --log_dir "$outdir" > "$out" 2>&1
if [ -f "$outdir/$rid.json" ]; then
  echo "[$(date +%H:%M:%S) gpu$gpu] done  $key $cname s$seed"
else
  echo "[$(date +%H:%M:%S) gpu$gpu] FAIL  $key $cname s$seed -> $out"
fi
ONEEOF
chmod +x "$LOGROOT/run_one.sh"

export GPU_POOL_ENV="$GPU_POOL"
export LOGROOT_ENV="$LOGROOT"
xargs -a "$QUEUE" -d '\n' -I{} -P "$NSLOTS" "$LOGROOT/run_one.sh" {}
echo "[all done] $(date)"
