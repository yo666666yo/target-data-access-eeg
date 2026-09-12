#!/usr/bin/env bash
# Reference-implementation check: braindecode EEGNetv4 in place of our own
# EEGNet, with the split, alignment, standardization, sampler, optimizer and
# stopping rule all unchanged.  The question is only whether the direction and
# rough size of the EA / SUP / SEL effects survive a different decoder.
set -u
ROOT=/home/USER/eeg-eval
mkdir -p $ROOT/logs/ref
Q=$ROOT/logs/ref/queue.txt
: > "$Q"
for ds in 2a 002 physionet; do
  case $ds in
    2a)        DSET=BNCI2014_001; SRC="--data-source official-gdf --official-gdf-root $ROOT/.cache/bciciv_2a" ;;
    002)       DSET=BNCI2014_002; SRC="" ;;
    physionet) DSET=PhysionetMI;  SRC="" ;;
  esac
  for cond in SRC EA SUP SEL; do
    case $cond in
      SRC) FLAGS="";                        A=0; T=0; NORM=outer_train_scaler ;;
      EA)  FLAGS="--euclidean_alignment";   A=1; T=0; NORM=outer_train_scaler ;;
      SUP) FLAGS="--target_supervised";     A=0; T=1; NORM=outer_train_scaler ;;
      SEL) FLAGS="--legacy_loso";           A=0; T=0; NORM=outer_train_scaler ;;
    esac
    [ "$ds" = "physionet" ] && [ "$cond" = "SEL" ] && continue
    SUF=""; [ "$cond" = "SEL" ] && SUF="-selection-heldout_subject"
    for s in 0 1 2; do
      RID="braindecode_equal_${DSET}_s${s}__cross-subject-loso-v4-align-${A}-targetsup-${T}-normalization-${NORM}${SUF}"
      echo "$ds|$cond|$s|$DSET|$SRC|$FLAGS|$NORM|$RID" >> "$Q"
    done
  done
done
echo "[plan] $(wc -l < "$Q") reference jobs"

cat > $ROOT/logs/ref/run_one.sh <<'INNER'
#!/usr/bin/env bash
IFS='|' read -r ds cond s dset src flags norm rid <<< "$1"
ROOT=/home/USER/eeg-eval
OUT=$ROOT/results/v4_ref_$ds
mkdir -p "$OUT"
[ -f "$OUT/$rid.json" ] && exit 0
G=$(( (RANDOM % 4) + 2 ))
LOG=$ROOT/logs/ref/${ds}_${cond}_s${s}.log
echo "[$(date +%H:%M:%S) gpu$G] start ref $ds $cond s$s"
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $ROOT/.venv/bin/python $ROOT/experiments/paper_runner.py \
  --variant braindecode-eegnet --weighting equal --dataset "$dset" $src \
  --normalization "$norm" $flags --seed "$s" --epochs 50 \
  --log_dir "$OUT" --run_id "$rid" > "$LOG" 2>&1 \
  && echo "[$(date +%H:%M:%S)] done  ref $ds $cond s$s" \
  || echo "[$(date +%H:%M:%S)] FAIL  ref $ds $cond s$s -> $LOG"
INNER
chmod +x $ROOT/logs/ref/run_one.sh
xargs -a "$Q" -d '\n' -I{} -P 8 $ROOT/logs/ref/run_one.sh {}
echo "[all done]"
