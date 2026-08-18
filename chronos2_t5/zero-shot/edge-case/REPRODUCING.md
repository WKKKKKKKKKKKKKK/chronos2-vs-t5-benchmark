# Reproducing the corruption-robustness results

Everything the paper claims, mapped to the command that produces it and the file it lands
in. Read sections 1 and 2, then either jump to section 3 to check a specific number
against its source, or run section 4 top to bottom to rebuild everything from scratch.

**Total cost from nothing: about 6 GPU-hours on one laptop GPU** (RTX 3500 Ada, 12.9 GB).
No cluster is needed and none was used. The study is inference-only over 46M- and
120M-parameter models at 100 series per dataset.

---

## 1. What this branch adds

`main` holds the completed internship deliverable: the Chronos-2 vs Chronos-T5 zero-shot
and one-shot benchmark, and the first corruption sweep. That state is tagged
`v1.0-internship`.

This branch adds the follow-up work behind the paper: multi-seed replication, direct
instrumentation of the Chronos-T5 tokeniser, a cross-learning sibling-shuffle control and
its small-dataset sensitivity check, a seventh corruption family with an
effect-size-matched control, a decode-only ablation that rules out sampling noise, and one
statistics module that every reported p-value must come from. It does not modify any result on `main` --
the edits to existing files are three registry lines, one `DECODE_SEED` global that
decouples Chronos-T5's decode from the corruption draw, and documentation.

## 2. Environment

```bash
conda env create -f ../../environment.yml     # chronos2_t5/environment.yml
conda activate chronos_bench
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # or CPU build
```

Run every script **from this directory** (`chronos2_t5/zero-shot/edge-case`). They resolve
the dataset registry by relative path and write into `results/`.

Absolute scores are not bit-identical across GPUs: Chronos-T5 draws 20 samples per
forecast and both models run in bf16. The headline quantity is a degradation *ratio*
against the same model's clean score from the same run, which is insulated from this. Do
not compare an absolute score produced here against one produced on other hardware.

## 3. Claim to evidence

Analysis unit is the **dataset** (n = 25) everywhere. Where several seeds exist they are
averaged *within* a dataset before any test, because seeds of one dataset are not
independent observations. WQL is the primary metric; MASE is reported alongside as a
robustness check, never as a second hypothesis.

### C1 -- Chronos-T5's error is decoupled from spike *magnitude*, not from spike *density*

WQL: Chronos-2 rho = +0.796 [+0.709, +0.867] on 98/100 curves; Chronos-T5 rho = -0.081
[-0.234, +0.068] on 45/100. Paired Wilcoxon **p = 2.98e-08** (Holm 8.94e-08), identical on
both metrics. Density axis: +0.952 vs **+0.674**. At maximum severity Chronos-T5 degrades
1.07x against Chronos-2's 1.50x -- the leaderboard inversion.

| | |
| --- | --- |
| produced by | `run_edge_cases.py` (seed 0) + `run_seeds.py --seeds 1,2,3` |
| raw | `results/edge_case_results.csv`, `results/edge_case_seeds.csv` |
| statistics | `statistics.py` -> `results/STATISTICS.md`, families `C1/spikes_intensity [WQL]` and `[MASE]`, `C1/spikes_density [...]` |
| narrative + figure | `analyse_seeds.py --metric WQL` -> `results/SEED_ANALYSIS_WQL.md`, `results/fig_seed_curves_WQL.png` |

The held-out recovery test (peak located on seed 0, tested on seeds 1-3) corroborates the
direction but is significant on MASE only (Holm 0.013) and not on WQL (Holm 0.076). It is
reported as suggestive, not as confirmation. Family `C1/held-out [...]` in `STATISTICS.md`.

### C2 -- the two natural mechanisms for the recovery are both false

Clamping does rise with severity in aggregate (0.0008 to 0.0045 of context points; share of
series with any clamped point 3.5% to 11.2%) and the mean scale inflates 1.8x. Neither
predicts *which* datasets recover:

| predictor of recovery | MASE | WQL | Holm |
| --- | --- | --- | --- |
| max clamped fraction | -0.279 | -0.321 | 0.316 |
| mean-scale inflation | -0.335 | -0.358 | 0.316 |
| growth of the excursion actually reaching the model | **+0.532** | **+0.545** | **0.031 / 0.029** |

Both suppression hypotheses carry the wrong sign; the one significant association runs
opposite to the blindness account.

| | |
| --- | --- |
| produced by | `measure_clamping.py` (CPU only -- tokeniser arithmetic, no forward pass) |
| raw | `results/clamping_measurements.csv` (1385 cells) |
| statistics | `statistics.py` -> family `C2/refutation` |
| narrative + figure | `analyse_clamping.py --metric WQL` -> `results/CLAMPING_ANALYSIS_spikes_intensity_WQL.md`, `results/fig_clamping_mechanism_spikes_intensity_WQL.png` |

### C2b -- the flat slope is not an artefact of the stochastic decode

Chronos-T5 draws 20 sample paths per forecast while Chronos-2 emits quantiles in one
deterministic pass, so sampling noise could in principle bury a real but shallow slope.
`DECODE_SEED` decouples the decode from the corruption draw, which makes this a genuine
single-variable manipulation: the corrupted contexts are fingerprinted and verified
byte-identical across runs, and only `torch.manual_seed` moves. Chronos-2 is not re-run --
a decode seed does nothing to a deterministic head.

| | rho vs spike magnitude | 95% CI | datasets positive |
| --- | --- | --- | --- |
| single decode seed | -0.019 | [-0.189, +0.155] | 11/25 |
| averaged over 8 decode seeds | +0.022 | [-0.168, +0.215] | 13/25 |

Averaging cuts sampler noise by about sqrt(8) and the slope stays indistinguishable from
zero (paired change over 25 datasets, p = 0.059). **Decode is excluded.** As a by-product,
the median s.d. across decode seeds alone is 0.043 against 0.065 across corruption *and*
decode seeds -- indicative only, since that run supplies 3 seeds against 8 here.

| | |
| --- | --- |
| produced by | `run_decode_ablation.py` (Chronos-T5 only, spike-magnitude family only) |
| raw | `results/decode_ablation.csv` (2200 rows, 8 decode seeds x 25 datasets) |
| narrative + figure | `analyse_decode_ablation.py --metric WQL` -> `results/DECODE_ABLATION_WQL.md`, `results/fig_decode_ablation_WQL.png` |

### C3 -- damage is governed by displacement *at* the forecast origin (PARTIAL)

Supported so far: the contrast between origin-pinned and randomly placed corruption in the
main sweep (`gap_boundary` vs `gap`), and E10's matched pair -- `drift` and `regime_trend`
displace the final context point identically at equal severity and degrade almost
identically (WQL 37.2 vs 38.8, p = 0.067), despite a breakpoint and a 4x steeper local
slope. So neither the breakpoint nor the slope is what matters.

**Not yet supported:** the controlled distance sweep (E5) that would turn this from a
comparison into a measured law. Do not write C3 as a law until that runs.

| | |
| --- | --- |
| raw | `results/edge_case_results.csv`, `results/edge_case_regime.csv` |
| narrative | `results/EDGE_CASE_REPORT.md`; Q2 in `results/REGIME_ANALYSIS_high_WQL.md` |

### C4 -- cross-learning needs a distribution-matched group

| group-mates | gain (gmean) | 95% CI | datasets helped |
| --- | --- | --- | --- |
| native (own dataset) | **1.045** | [1.012, 1.085] | 18/25, p = 0.022 |
| foreign, same frequency | 1.020 | [0.979, 1.058] | -- |
| foreign, different frequency | 1.006 | [0.966, 1.044] | -- |

Only the native group clears 1, so the benefit is not generic pooling. Same- versus
different-frequency is significant on WQL (Holm 0.030) but not on MASE (0.15), so frequency
matching recovers *part* of it and the metric dependence must be stated. Two datasets are
absent from the same-frequency arm by construction: `monash_australian_electricity`
(subhourly only) and `monash_traffic` (pool held 8 series, below the 99 required). Both are
reported in the analysis output rather than dropped silently.

**The gain is not evenly spread, and the table above must not be quoted alone.** Four
datasets hold fewer than the 100 series a group can take (`monash_australian_electricity`
5, `ercot` 8, `exchange_rate` 8, `monash_cif_2016` 72). A target grouped with its own
dataset gets a group of whatever size that dataset has, while both foreign conditions are
always filled to 100, so for these four the contrast varies group size as well as group
membership. Restricting to the 21 full-size datasets moves the two halves of C4 in
*opposite* directions:

| | 25 datasets | 21 full-size datasets |
| --- | --- | --- |
| native gain, WQL | 1.045 [1.012, 1.085], 18/25, p = 0.022 | 1.032 [1.002, 1.067], 14/21, p = 0.095 |
| native gain, MASE | 1.022 [1.003, 1.043] | **1.018 [0.997, 1.038] -- CI no longer clears 1** |
| same- vs different-frequency, Holm | 0.030 (WQL) / 0.150 (MASE) | **0.0004 (WQL) / 0.058 (MASE)** |

So "only the native group clears 1" survives on WQL but not on MASE once the four are
removed, while the frequency contrast gets sharply stronger, because the
different-frequency gain falls below 1 (0.986 WQL) without them. Report both columns.
Dropping the four *lowers* the native gain even though their groups are the smallest.
A draft explained that by within-collection homogeneity; `measure_homogeneity.py` tested
the explanation and **refuted** it -- `monash_cif_2016` carries almost the whole effect
(own-dataset gain 1.395, the largest in the suite) yet ranks 23rd of 25 for internal
correlation. The paper therefore reports the phenomenon, and the fact that C4's point
estimate leans on one collection, without a mechanism.

| | |
| --- | --- |
| produced by | `run_cl_shuffle.py` |
| raw | `results/crosslearning_shuffle.csv` |
| narrative + figure | `analyse_cl_shuffle.py --metric WQL` -> `results/CL_SHUFFLE_ANALYSIS_WQL.md`, `results/fig_cl_shuffle_WQL.png` |
| sensitivity | `analyse_cl_sensitivity.py` -> `results/CL_SENSITIVITY_ANALYSIS.md` (both metrics, both subsets, one file) |
| refuted explanation | `measure_homogeneity.py` -> `results/HOMOGENEITY.md`. Supports no claim; it is the record of one that was withdrawn |

### C5 -- the undersensitivity is specific to observation-level magnitude

On `regime_trend` both models track severity essentially perfectly (WQL rho 0.999 / 0.998;
MASE exactly 1.000 for both, which makes the paired test degenerate rather than
significant). Restricted to severities costing the same 1.02-1.55x as the spike range:

| family | model | WQL rho | MASE rho |
| --- | --- | --- | --- |
| regime, effect-matched | Chronos-2 | +0.74 [+0.60, +0.86] | +0.89 [+0.82, +0.95] |
| regime, effect-matched | Chronos-T5 | **+0.60 [+0.47, +0.73]** | **+0.71 [+0.57, +0.83]** |
| spike magnitude | Chronos-2 | +0.78 [+0.69, +0.86] | +0.83 |
| spike magnitude | Chronos-T5 | **-0.10 [-0.25, +0.03]** | **-0.11** |

This is the paper's **positive control**: the same code path that reads -0.10 on spike
magnitude reads +0.60 here at matched damage, so the flat slope is neither a broken
severity axis nor a small-effect artefact.

Qualification that must travel with it: on the low grid as a whole Chronos-2's slope is
significantly higher than Chronos-T5's (WQL 0.749 vs 0.662, p = 0.0045; MASE 0.870 vs
0.701, p = 0.0014). Chronos-T5 is mildly less responsive in general -- a difference of
degree. Only on spike magnitude is it a difference of kind.

| | |
| --- | --- |
| produced by | `run_seeds.py --families regime_trend` on two grids |
| raw | `results/edge_case_regime.csv` (coarse), `results/edge_case_regime_low.csv` (effect-matched) |
| narrative + figure | `analyse_regime.py --csv <file> --metric WQL` -> `results/REGIME_ANALYSIS_{high,low}_WQL.md`, `results/fig_regime_{high,low}_WQL.png` |

---

## 4. Run order

Steps 1-6 are independent of one another; step 7 reads their outputs. Every runner appends
per `(seed, dataset)` and skips what is already present, so any of them can be killed and
restarted.

Timings marked *(measured)* are wall-clock from the actual runs on the machine described
above. The one marked *(estimated)* is extrapolated from the measured per-cell cost,
because that sweep predates timing being recorded. Step 6 is likewise extrapolated, from
its Chronos-T5 row count against the measured step 2.

```bash
conda activate chronos_bench      # run everything from chronos2_t5/zero-shot/edge-case

# 1. main sweep, seed 0, six families                GPU  ~75 min (estimated)
python run_edge_cases.py
#    -> results/edge_case_results.csv, EDGE_CASE_REPORT.md, examples/, fig_*_curves.png

# 2. multi-seed replication of the spike families    GPU  ~70 min (measured)
python run_seeds.py --seeds 1,2,3
#    -> results/edge_case_seeds.csv

# 3. tokeniser instrumentation                       CPU   ~4 min (measured)
python measure_clamping.py
#    -> results/clamping_measurements.csv

# 4. cross-learning sibling shuffle                  GPU  ~55 min (measured)
python run_cl_shuffle.py
#    -> results/crosslearning_shuffle.csv

# 5. seventh family, two severity grids         GPU  ~46 + ~38 min (measured)
python run_seeds.py --families regime_trend --seeds 0,1,2,3 \
       --out edge_case_regime.csv
python run_seeds.py --families regime_trend --seeds 0,1,2,3 \
       --severities 0.01,0.02,0.04,0.06,0.09,0.13,0.18,0.25 \
       --out edge_case_regime_low.csv

# 6. decode-only ablation, Chronos-T5 x 8 decode seeds   GPU  ~96 min (measured)
python run_decode_ablation.py
#    -> results/decode_ablation.csv

# 7. analysis -- no GPU, seconds each
python statistics.py                                  # -> results/STATISTICS.md
for m in WQL MASE; do
  python analyse_seeds.py      --metric $m
  python analyse_clamping.py   --metric $m
  python analyse_cl_shuffle.py --metric $m
  python analyse_regime.py --csv edge_case_regime.csv     --metric $m
  python analyse_regime.py --csv edge_case_regime_low.csv --metric $m
  python analyse_decode_ablation.py --metric $m
done
python analyse_cl_sensitivity.py     # -> results/CL_SENSITIVITY_ANALYSIS.md (both metrics)
python measure_homogeneity.py        # -> results/HOMOGENEITY.md  (~2 min, reads the registry)
python mk_fig_suite.py               # -> results/fig_corruption_suite.{png,pdf}
```

## 5. Eight ways to get this silently wrong

Each of these produced a wrong answer at some point during the study. They are listed
because none of them raises an error -- they all return plausible numbers.

1. **The resume key is `(seed, dataset)` and does not include the family.** Running a new
   family into an existing CSV skips every cell as "already done" and writes nothing.
   Always pass `--out` for a new family. This is why the regime runs have their own files.

2. **`statistics.py` used to hardcode `metric="MASE"` for C1** while the study declares WQL
   primary, which quietly made every headline number the secondary metric's. It now runs
   both. If you add a claim, add both metrics.

3. **Seeds are not independent observations.** Treating the 100 `(seed, dataset)` curves as
   independent turns p = 2.98e-08 into p = 3.2e-32, and p = 0.038 into p = 9.5e-4. Average
   within dataset first. `statistics.py` does; ad-hoc analysis does not.

4. **`analyse_regime.py` outputs are tagged by source CSV.** Before that fix, analysing the
   low-severity run silently overwrote the high-severity figures and reports.

5. **Severity participates in the corruption seed.** Every point on a severity curve is an
   independent draw, deliberately, so that nested spike sets cannot manufacture false
   monotonicity. It also means a single-seed curve cannot separate shape from position
   luck, which is why step 2 is mandatory rather than optional.

6. **`results/clamping_measurements_smoke.csv` is not data.** It is a 4-dataset trial run,
   left untracked on purpose. Do not analyse it.

7. **A killed run leaves rows behind, and resume will not notice.** The decode ablation was
   smoke-tested, the process was killed on a timeout after it had already written cells,
   and the full run then appended the same cells again -- 2244 rows where 2200 were
   expected. The duplicates were byte-identical so dropping them was lossless, but nothing
   warned about it. After any interrupted run, check the row count against
   `seeds x datasets x conditions` before analysing, and verify duplicates agree before
   de-duplicating.

8. **Aligning series to the shortest member of a collection destroys the collection.** The
   first version of `measure_homogeneity.py` truncated every series in a dataset to the
   length of its shortest, so one short series collapsed the usable window for all of them:
   six datasets returned NaN -- including the one that decided the question -- and a yearly
   collection returned no usable series at all. It raised no error. Alignment is now per
   pair, so a short series only costs the pairs it appears in.

## 6. Naming

Four corruption functions produce seven swept families, and the example figures use a
second set of names for four of them. Ten names, seven things. The mapping is fixed in the
`perturbations.py` module docstring and reproduced here:

| sweep / results CSV | example figures | paper |
| --- | --- | --- |
| `spikes_intensity` | `spikes_intensity` | spike magnitude |
| `spikes_density` | `spikes_density` | spike density |
| `drift` | `drift_ramp` | global drift |
| `drift_step` | `level_shift` | transient level shift |
| `gap` | `missing_random` | dropout (random position) |
| `gap_boundary` | `missing_boundary` | dropout (at forecast origin) |
| `regime_trend` | -- | persistent trend change |

Do not introduce an eleventh name.
