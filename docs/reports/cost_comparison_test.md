# Cost/latency benchmark (EPIC G4)
- Dataset: `test`

| Model | Precision | Recall | Latency (ms) | Speedup vs Claude | Cost $/1M | Savings vs Claude |
| --- | --- | --- | --- | --- | --- | --- |
| claude | 1.000 | 1.000 | 1500.00 | 1.000 | 1800.00 | 1.000 |
| gbdt | 0.691 | 0.844 | 0.40 | 3758.966 | 0.05 | 33063.944 |
| xgb | 0.739 | 0.756 | 0.18 | 8215.499 | 0.02 | 89281.045 |
| rf | 0.765 | 0.867 | 0.79 | 1887.892 | 0.07 | 26781.105 |
| text | 0.930 | 0.889 | 0.04 | 36922.598 | 0.00 | 408396.344 |
| st | 0.930 | 0.889 | 6.90 | 217.383 | 2.28 | 789.120 |
| ensemble_gbdt_tfidf | 0.691 | 0.844 | 0.13 | 11794.701 | 0.01 | 178751.585 |
| ensemble_gbdt_st | 0.691 | 0.844 | 1.10 | 1363.678 | 0.85 | 2112.148 |
| ensemble_xgb_tfidf | 0.784 | 0.889 | 0.23 | 6650.987 | 0.02 | 73684.379 |
| ensemble_xgb_st | 0.765 | 0.867 | 1.16 | 1290.751 | 0.89 | 2018.490 |
| ensemble_rf_tfidf | 0.870 | 0.889 | 0.67 | 2244.915 | 0.07 | 25852.109 |
| ensemble_rf_st | 0.851 | 0.889 | 1.60 | 935.474 | 1.43 | 1261.193 |
| llm | 0.796 | 0.956 | 5245.23 | 0.286 | 79.84 | 22.545 |

- **Best recall**: `llm` hits 0.956 recall / 0.796 precision on `test`.
- **Fastest**: `text` is 36922.598× quicker than Claude.
- **Cheapest**: `text` delivers 408396.344× lower cost per alert.

_Claude baseline metrics include the $0.0018/alert estimate from the assignment brief; local costs come from `configs/costs.json`._