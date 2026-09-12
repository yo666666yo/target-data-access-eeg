# Target-Data Access and Aggregation Effects in Cross-Subject EEG Evaluation

Code, run logs and per-trial predictions for the preprint of the same name.
The manuscript has not been submitted to or accepted at any venue.

Cross-subject EEG decoders are compared across protocols that use the held-out
subject differently, and the accuracy gaps between them get read as the value of
that access. That reading only holds if everything else is matched. This
repository contains the experiments that check what happens when it is, and the
logs those checks produced.

## What is here

```
experiments/     the runner, the schema gate, the data sources, the control audit
analysis/        every statistic reported in the paper
figures/         the figure and table generators
scripts/         the launch scripts that produced the logs in results/
tests/           73 tests, mostly guards against ways this study has already gone wrong
results/         182 run logs with per-trial predictions, plus the supplements
manuscript/      LaTeX source and the compiled PDF
```

One edit has been made to the released logs: the absolute paths they record
under `provenance` ran from a home directory on the lab machine, and every
occurrence of that directory has been replaced by `/home/USER`. Nothing else in
the logs was altered, and the same substitution has been applied to the launch
scripts. The source-snapshot hashes cover the source files, not the log, so
they are unaffected.

## Reproducing a number from the paper

Every figure and table is generated, never typed. With the logs in `results/`:

```bash
python analysis/build_paper_evidence.py     # stats -> tables -> figures
```

Individual analyses:

```bash
python analysis/v4_stats.py "BCI IV 2a=results/v4_2a/*.json" ...   # main grid
python analysis/dose_stats.py                                      # calibration budget
python analysis/reference_check.py                                 # braindecode decoder
python analysis/session_split.py                                   # chronological split
python analysis/variance_table.py                                  # per-run variance
```

## Re-running the experiments

```bash
python experiments/paper_runner.py --variant eegnet --weighting equal \
    --dataset BNCI2014_001 --data-source official-gdf \
    --official-gdf-root <path to the BCI IV 2a archives> \
    --normalization outer_train_scaler --seed 0 --epochs 50 \
    --log_dir results/v4_2a
```

Add `--euclidean_alignment` for the aligned arm, `--target_supervised` for the
supervised one, `--available_trials N` to cap the calibration budget,
`--target_split session` for the chronological split on BCI IV 2a, and
`--variant braindecode-eegnet` for the reference decoder.

Every log records the exact command that produced it under
`provenance.command_text`. The scripts that launched the sweeps are in
`scripts/`: `run_v4_matrix.sh` for the main grid, `run_dose.sh` and
`run_dose_ea.sh` for the calibration budget, `run_ref.sh` for the reference
decoder and `run_sess.sh` for the chronological split. They assume a machine
with several GPUs and expect `$PROJECT_ROOT` to be edited to wherever the
repository sits.

Before trusting a comparison, replay the controls against the real data:

```bash
python experiments/verify_controls.py --dataset BNCI2014_001 \
    --data-source official-gdf --official-gdf-root <path>
```

It rebuilds each condition's folds and fails if the source partition, the epoch
budget, the target sampling rate or the reserved test trials differ between
arms. It is meant to run *before* the experiments. Pass `--available_trials N`
or `--target_split session` to audit those arms under the same checks.

## Source snapshots

Logs carry a hash over the source files that produced them. Three snapshots
appear, because two additions required changing the runner after the main grid
had already been run:

| snapshot | runs |
|---|---|
| `ce80d930…` | the main 2×2 grid, `SEL` and `POOL` |
| `53aca1d1…` | the capped calibration budget (`--available_trials`) |
| `9ba6f6f2…` | the braindecode reference decoder |

A fourth covers the session-split runs. Comparisons in the paper are made
within a snapshot wherever the contrast depends on it.

HEAD is the final state of the tooling and does not match every snapshot: the
control auditor and the provenance collector were extended after the runs, so
that the auditor covers the capped-budget and chronological arms and so that
`braindecode` and `mne` versions are recorded. Neither change touches the
training path. Where HEAD and a log disagree, the per-file hashes inside the
log are what that run actually used.

## Data

The three datasets are public and are not redistributed here.

- **BCI Competition IV 2a** — obtain the official GDF archives and true-label
  archive, then pass the directory to `--official-gdf-root`.
- **BNCI2014-002** and **PhysionetMI** — fetched through MOABB on first use.

`results/` contains per-trial predictions and labels on the reserved test half,
which is what lets the aggregation analysis be reproduced without retraining. It
contains no raw recordings.

## Environment

`results/reference_env.txt` is a full `pip freeze` from the machine that ran the
reference-decoder experiments. The versions that matter:

```
torch==2.6.0+cu124   numpy==2.2.6   scikit-learn==1.7.2
moabb==1.5.0         mne==1.12.1    braindecode==0.8.1
```

braindecode 0.8.1 imports pre-1.0 MOABB dataset names at package-import time.
`paper_runner.py` aliases the three affected names before importing the model;
MOABB itself is untouched and the model code is upstream's. braindecode 1.0.0
does not work here: it pulls a torchaudio built against CUDA 13, which will not
load beside torch 2.6.0+cu124.

## Supplementary material

- `results/variance_table.md` — per-run variance table at full precision, with
  the binomial bound each noise term must satisfy
- `results/supplement_budget_curve.txt` — the full calibration-budget curve
- `results/reference_env.txt` — environment of the reference-decoder runs

## Licence

MIT, see `LICENSE`.
