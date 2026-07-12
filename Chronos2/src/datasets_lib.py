"""Dataset registry for the Chronos-2 zero-shot study.

This is intentionally a *verbatim copy* of the registry used by the sibling
Chronos-T5 reproduction (`Chronos_benchmark/src/datasets_lib.py`): the same 25
Benchmark II datasets, the same per-dataset horizons (paper Table 3), the same
deterministic `MAX_SERIES` cap and the same HF repo. Keeping it byte-for-byte
identical guarantees that the Chronos-2 numbers produced here are evaluated on
*exactly* the same series, in the same order, with the same held-out windows as
the Chronos-T5 baseline — so the two are directly comparable.

The datasets are 25 of the paper's 27 Benchmark II datasets, available in the
official `autogluon/chronos_datasets` HuggingFace repo (the two ETT datasets are
absent from that repo).

Seasonality is NOT stored here: gluonts computes the MASE seasonality at eval time
from each series' frequency, identical to the official chronos `evaluate.py`.
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