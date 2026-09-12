#!/usr/bin/env bash
# EA at the capped budget.  Needed because the whitener is estimated from the
# available half, so EA at |C_s|=45 is a different arm from EA at the full
# half -- without it there is no matched-budget SUP-minus-EA contrast.
set -u
ROOT=/home/USER/eeg-eval
mkdir -p $ROOT/logs/dose
Q=$ROOT/logs/dose/queue_ea.txt
: > "$Q"
for ds in 2a 002; do
  case $ds in
    2a)  DSET=BNCI2014_001; SRC="--data-source official-gdf --official-gdf-root $ROOT/.cache/bciciv_2a"; CAPS="45 144" ;;
    002) DSET=BNCI2014_002; SRC=""; CAPS="45" ;;
  esac
  for cap in $CAPS; do
    for s in 0 1 2; do
      RID="eegnet_equal_${DSET}_s${s}_cap${cap}__cross-subject-loso-v4-align-1-targetsup-0-normalization-outer_train_scaler"
      echo "$ds|EA|$cap|$s|$DSET|$SRC|--euclidean_alignment|$RID" >> "$Q"
    done
  done
done
echo "[plan] $(wc -l < "$Q") EA dose jobs"
xargs -a "$Q" -d '\n' -I{} -P 3 $ROOT/logs/dose/run_one.sh {}
echo "[ea done]"
