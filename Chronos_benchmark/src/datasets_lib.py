"""Dataset registry for the Chronos paper's Benchmark II (zero-shot) reproduction.

This is just the registry — the three constants below. The evaluation/fine-tuning
scripts (`run_zeroshot_official.py`, `finetune_oneshot.py`, `run_oneshot_official.py`,
`make_notebook.py`) import them and do their own loading via gluonts (see
`run_zeroshot_official.to_gluonts_univariate`). There is no loader here.

The datasets are 25 of the paper's 27 Benchmark II datasets, available in the
official `autogluon/chronos_datasets` HuggingFace repo (the two ETT datasets are
absent from that repo).

Faithfulness to the paper (Ansari et al., 2024, "Chronos: Learning the Language
of Time Series", TMLR):
  - horizon H per dataset = the paper's prediction length (Table 3);
  - one held-out last-H window per series (the eval scripts use num_windows = 1),
    exactly as the paper;
  - MAX_SERIES caps each dataset (deterministic evenly-spaced subsample) for
    laptop tractability — metric *definitions* are unchanged.

Seasonality is NOT stored here: gluonts computes the MASE seasonality at eval time
from each series' frequency (`get_seasonality(start.freqstr)`; daily->1, hourly->24,
business-daily->5, monthly->12, ...). That freq-inferred value is the single source
of truth, identical to what the official chronos `evaluate.py` uses.
"""

# (config, horizon) for each Benchmark II dataset. horizon = paper Table 3.
BENCHMARK_II = [
    ("monash_australian_electricity", 48),  # energy, 30-min
    ("monash_cif_2016",               12),  # banking, monthly
    ("monash_car_parts",              12),  # retail (intermittent), monthly
    ("monash_covid_deaths",           30),  # healthcare, daily
    ("dominick",                       8),  # retail, weekly
    ("ercot",                         24),  # energy, hourly
    ("exchange_rate",                 30),  # finance, business-daily
    ("monash_fred_md",                12),  # economics, monthly
    ("monash_hospital",               12),  # healthcare, monthly
    ("monash_m1_monthly",             18),  # M1, monthly
    ("monash_m1_quarterly",            8),  # M1, quarterly
    ("monash_m1_yearly",               6),  # M1, yearly
    ("monash_m3_monthly",             18),  # M3, monthly
    ("monash_m3_quarterly",            8),  # M3, quarterly
    ("monash_m3_yearly",               6),  # M3, yearly
    ("m4_quarterly",                   8),  # M4, quarterly
    ("m4_yearly",                      6),  # M4, yearly
    ("m5",                            28),  # retail (Walmart), daily
    ("nn5",                           56),  # finance (ATM), daily
    ("monash_nn5_weekly",              8),  # finance (ATM), weekly
    ("monash_tourism_monthly",        24),  # tourism, monthly
    ("monash_tourism_quarterly",       8),  # tourism, quarterly
    ("monash_tourism_yearly",          4),  # tourism, yearly
    ("monash_traffic",                24),  # transport, hourly
    ("monash_weather",                30),  # nature, daily
]
MAX_SERIES = 1000
HF_REPO = "autogluon/chronos_datasets"
