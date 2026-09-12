#!/usr/bin/env bash
# Matched-calibration-budget arm: cap C_s so that |C_s| is equal across
# datasets, holding the reserved test half fixed.  If the EA-vs-SUP ordering
# still reverses at a common budget, calibration size and the repetition it
# implies are not what produced the reversal.
set -u
ROOT=/home/USER/eeg-eval
PY=$ROOT/.venv/bin/python
GPUS=(3 4 5 3 4 5)
mkdir -p $ROOT/logs/dose

QUEUE=$ROOT/logs/dose/queue.txt
: > "$QUEUE"
for ds in 2a 002; do
  case $ds in
    2a)  DSET=BNCI2014_001; SRC="--data-source official-gdf --official-gdf-root $ROOT/.cache/bciciv_2a"; CAPS="45 144" ;;
    002) DSET=BNCI2014_002; SRC=""; CAPS="45" ;;
  esac
  for cap in $CAPS; do
    for cond in SUP EASUP; do
      case $cond in
        SUP)   FLAGS="--target_supervised" ;;
        EASUP) FLAGS="--euclidean_alignment --target_supervised" ;;
      esac
      A=0; [ "$cond" = "EASUP" ] && A=1
      for s in 0 1 2; do
        RID="eegnet_equal_${DSET}_s${s}_cap${cap}__cross-subject-loso-v4-align-${A}-targetsup-1-normalization-outer_train_scaler"
        echo "$ds|$cond|$cap|$s|$DSET|$SRC|$FLAGS|$RID" >> "$QUEUE"
      done
    done
  done
done
echo "[plan] $(wc -l < "$QUEUE") dose jobs"

cat > $ROOT/logs/dose/run_one.sh <<'INNER'
#!/usr/bin/env bash
IFS='|' read -r ds cond cap s dset src flags rid <<< "$1"
ROOT=/home/USER/eeg-eval
OUT=$ROOT/results/v4_dose_$ds
mkdir -p "$OUT"
[ -f "$OUT/$rid.json" ] && exit 0
G=$(( (RANDOM % 3) + 3 ))
LOG=$ROOT/logs/dose/${ds}_${cond}_cap${cap}_s${s}.log
echo "[$(date +%H:%M:%S) gpu$G] start $ds $cond cap$cap s$s"
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $ROOT/.venv/bin/python $ROOT/experiments/paper_runner.py \
  --variant eegnet --weighting equal --dataset "$dset" $src \
  --normalization outer_train_scaler $flags --available_trials "$cap" \
  --seed "$s" --epochs 50 --log_dir "$OUT" --run_id "$rid" > "$LOG" 2>&1 \
  && echo "[$(date +%H:%M:%S)] done  $ds $cond cap$cap s$s" \
  || echo "[$(date +%H:%M:%S)] FAIL  $ds $cond cap$cap s$s -> $LOG"
INNER
chmod +x $ROOT/logs/dose/run_one.sh
xargs -a "$QUEUE" -d '\n' -I{} -P 4 $ROOT/logs/dose/run_one.sh {}
echo "[all done]"
