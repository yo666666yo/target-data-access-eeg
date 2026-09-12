#!/usr/bin/env bash
# Session-aware split on BCI IV 2a: the available half is the earlier recording
# session and the reserved half the later one.  Same 288/288 budget as the
# random split, so the only thing that changes is whether the partition is
# arbitrary or chronological.  Five seeds, since the reviews asked for more
# repetitions and this split is deterministic in its partition -- only the
# initialization and the source draw vary.
set -u
ROOT=/home/USER/eeg-eval
mkdir -p $ROOT/logs/sess
Q=$ROOT/logs/sess/queue.txt
: > "$Q"
DSET=BNCI2014_001
SRC="--data-source official-gdf --official-gdf-root $ROOT/.cache/bciciv_2a"
for cond in SRC EA SUP EASUP; do
  case $cond in
    SRC)   FLAGS="";                                      A=0; T=0 ;;
    EA)    FLAGS="--euclidean_alignment";                 A=1; T=0 ;;
    SUP)   FLAGS="--target_supervised";                   A=0; T=1 ;;
    EASUP) FLAGS="--euclidean_alignment --target_supervised"; A=1; T=1 ;;
  esac
  for s in 0 1 2 3 4; do
    RID="eegnet_equal_${DSET}_s${s}_sess__cross-subject-loso-v4-align-${A}-targetsup-${T}-normalization-outer_train_scaler"
    echo "$cond|$s|$DSET|$SRC|$FLAGS|$RID" >> "$Q"
  done
done
echo "[plan] $(wc -l < "$Q") session-split jobs"

cat > $ROOT/logs/sess/run_one.sh <<'INNER'
#!/usr/bin/env bash
IFS='|' read -r cond s dset src flags rid <<< "$1"
ROOT=/home/USER/eeg-eval
OUT=$ROOT/results/v4_sess_2a
mkdir -p "$OUT"
[ -f "$OUT/$rid.json" ] && exit 0
G=$(( (RANDOM % 4) + 2 ))
LOG=$ROOT/logs/sess/${cond}_s${s}.log
echo "[$(date +%H:%M:%S) gpu$G] start sess $cond s$s"
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $ROOT/.venv/bin/python $ROOT/experiments/paper_runner.py \
  --variant eegnet --weighting equal --dataset "$dset" $src \
  --normalization outer_train_scaler --target_split session $flags \
  --seed "$s" --epochs 50 --log_dir "$OUT" --run_id "$rid" > "$LOG" 2>&1 \
  && echo "[$(date +%H:%M:%S)] done  sess $cond s$s" \
  || echo "[$(date +%H:%M:%S)] FAIL  sess $cond s$s -> $LOG"
INNER
chmod +x $ROOT/logs/sess/run_one.sh
xargs -a "$Q" -d '\n' -I{} -P 8 $ROOT/logs/sess/run_one.sh {}
echo "[all done]"
