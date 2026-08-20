# ⚾ MLB Target Intelligence Suite

An automated, quantitative MLB intelligence pipeline that evaluates daily matchups across six independent statistical engines. The system ingests official MLB Stats API feeds, models matchup dynamics via discrete probability distributions and survival analysis, and outputs dark-mode visual target boards and structured datasets.

---

## 🏛️ System Architecture

```text
├── core/
│   ├── data_loader.py            # Unified single-pass API caching & slate ingestion
│   └── settlement_engine.py      # Immutable Player-ID boxscore verification
├── models/
│   ├── model_hr.py               # Home Run Intensity Engine (NHPP)
│   ├── model_hits.py             # Contact & Hit Engine (1+ / 2+ Hits)
│   ├── model_total_bases.py      # Extra-Base Slugging Model (2+ TB)
│   ├── model_hr_rbi.py           # Production Combo Model (1.5+ H+R+RBI)
│   ├── model_pitcher_ks.py       # Pitcher Strikeout Model (Whiff / BF Poisson)
│   └── model_weibull.py          # Parametric Drought Survival Model (MLE)
├── exports/                      # Segregated daily output artifacts
│   ├── hr/
│   ├── hits/
│   ├── total_bases/
│   ├── hr_rbi/
│   ├── pitcher_ks/
│   └── weibull/
├── requirements.txt
└── run_models.py                 # Master CLI execution entry point
