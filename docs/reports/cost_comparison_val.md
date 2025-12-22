# Cost/latency benchmark (EPIC G4)
- Dataset: `val`

| Model | Precision | Recall | Latency (ms) | Speedup vs Claude | Cost $/1M | Savings vs Claude |
| --- | --- | --- | --- | --- | --- | --- |
| claude | 1.000 | 1.000 | 1500.00 | 1.000 | 1800.00 | 1.000 |
| gbdt | 0.700 | 0.824 | 1.43 | 1049.649 | 0.05 | 33063.944 |
| xgb | 0.688 | 0.647 | 0.53 | 2834.319 | 0.02 | 89281.045 |
| rf | 0.737 | 0.824 | 1.76 | 850.194 | 0.07 | 26781.105 |
| text | 0.933 | 0.824 | 0.12 | 12964.963 | 0.00 | 408396.344 |
| st | 0.933 | 0.824 | 5.71 | 262.518 | 2.28 | 789.120 |
| ensemble_gbdt_tfidf | 0.824 | 0.824 | 0.26 | 5674.654 | 0.01 | 178751.585 |
| ensemble_gbdt_st | 0.778 | 0.824 | 2.13 | 702.653 | 0.85 | 2112.148 |
| ensemble_xgb_tfidf | 0.824 | 0.824 | 0.64 | 2339.187 | 0.02 | 73684.379 |
| ensemble_xgb_st | 0.737 | 0.824 | 2.23 | 671.495 | 0.89 | 2018.490 |
| ensemble_rf_tfidf | 0.875 | 0.824 | 1.83 | 820.702 | 0.07 | 25852.109 |
| ensemble_rf_st | 0.875 | 0.824 | 3.58 | 419.564 | 1.43 | 1261.193 |
| llm | 0.750 | 0.882 | 200.00 | 7.500 | 79.84 | 22.545 |

- **Best recall**: `llm` hits 0.882 recall / 0.750 precision on `val`.
- **Fastest**: `text` is 12964.963× quicker than Claude.
- **Cheapest**: `text` delivers 408396.344× lower cost per alert.

_Claude baseline metrics include the $0.0018/alert estimate from the assignment brief; local costs come from `configs/costs.json`._