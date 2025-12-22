# Model Comparison Dashboard

- Dataset: `val`
- Source: `artifacts/eval/val_comparison_summary.json`

| Model | Accuracy | P(label=1) | R(label=1) | F1(label=1) | ROC AUC | Avg Precision | Latency (ms/sample) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gbdt | 0.679 | 0.700 | 0.824 | 0.757 | 0.706 | 0.813 | 1.43 |
| xgb | 0.607 | 0.688 | 0.647 | 0.667 | 0.684 | 0.801 | 0.53 |
| rf | 0.714 | 0.737 | 0.824 | 0.778 | 0.722 | 0.833 | 1.76 |
| text | 0.857 | 0.933 | 0.824 | 0.875 | 0.914 | 0.947 | 0.12 |
| st | 0.857 | 0.933 | 0.824 | 0.875 | 0.888 | 0.939 | 5.71 |
| ensemble_gbdt_tfidf | 0.786 | 0.824 | 0.824 | 0.824 | 0.840 | 0.897 | 0.26 |
| ensemble_gbdt_st | 0.750 | 0.778 | 0.824 | 0.800 | 0.834 | 0.903 | 2.13 |
| ensemble_xgb_tfidf | 0.786 | 0.824 | 0.824 | 0.824 | 0.813 | 0.879 | 0.64 |
| ensemble_xgb_st | 0.714 | 0.737 | 0.824 | 0.778 | 0.802 | 0.883 | 2.23 |
| ensemble_rf_tfidf | 0.821 | 0.875 | 0.824 | 0.848 | 0.856 | 0.903 | 1.83 |
| ensemble_rf_st | 0.821 | 0.875 | 0.824 | 0.848 | 0.834 | 0.908 | 3.58 |
| llm | 0.750 | 0.750 | 0.882 | 0.811 | 0.714 | 0.733 | 200.00 |

_Note: Latency values for models marked as 0 were not measured during this run and may rely on cached predictions (e.g., the LLM reasoner)._


## Detailed metrics & threshold analysis


### gbdt (n=28)
- Accuracy: 0.679 | ROC-AUC: 0.706 | Average Precision: 0.813
- Label=1 P/R/F1: 0.700 / 0.824 / 0.757
- Macro avg P/R/F1: 0.662 / 0.639 / 0.642
- Weighted avg P/R/F1: 0.671 / 0.679 / 0.666
- Latency: 1.43 ms/sample (total 0.040 s)
Plots:
![gbdt_roc_curve.png](gbdt_roc_curve.png)
![gbdt_pr_curve.png](gbdt_pr_curve.png)
![gbdt_prob_distribution.png](gbdt_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.625 | 0.882 | 15 | 9 | 2 | 2 |
| 0.200 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.300 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.400 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.500 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.600 | 0.737 | 0.824 | 14 | 5 | 6 | 3 |
| 0.700 | 0.812 | 0.765 | 13 | 3 | 8 | 4 |
| 0.800 | 0.769 | 0.588 | 10 | 3 | 8 | 7 |
| 0.900 | 0.750 | 0.529 | 9 | 3 | 8 | 8 |

### xgb (n=28)
- Accuracy: 0.607 | ROC-AUC: 0.684 | Average Precision: 0.801
- Label=1 P/R/F1: 0.688 / 0.647 / 0.667
- Macro avg P/R/F1: 0.594 / 0.596 / 0.594
- Weighted avg P/R/F1: 0.614 / 0.607 / 0.610
- Latency: 0.53 ms/sample (total 0.015 s)
Plots:
![xgb_roc_curve.png](xgb_roc_curve.png)
![xgb_pr_curve.png](xgb_pr_curve.png)
![xgb_prob_distribution.png](xgb_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.714 | 0.882 | 15 | 6 | 5 | 2 |
| 0.200 | 0.714 | 0.882 | 15 | 6 | 5 | 2 |
| 0.300 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.400 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.500 | 0.688 | 0.647 | 11 | 5 | 6 | 6 |
| 0.600 | 0.786 | 0.647 | 11 | 3 | 8 | 6 |
| 0.700 | 0.786 | 0.647 | 11 | 3 | 8 | 6 |
| 0.800 | 0.769 | 0.588 | 10 | 3 | 8 | 7 |
| 0.900 | 0.750 | 0.529 | 9 | 3 | 8 | 8 |

### rf (n=28)
- Accuracy: 0.714 | ROC-AUC: 0.722 | Average Precision: 0.833
- Label=1 P/R/F1: 0.737 / 0.824 / 0.778
- Macro avg P/R/F1: 0.702 / 0.684 / 0.689
- Weighted avg P/R/F1: 0.709 / 0.714 / 0.708
- Latency: 1.76 ms/sample (total 0.049 s)
Plots:
![rf_roc_curve.png](rf_roc_curve.png)
![rf_pr_curve.png](rf_pr_curve.png)
![rf_prob_distribution.png](rf_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.593 | 0.941 | 16 | 11 | 0 | 1 |
| 0.200 | 0.682 | 0.882 | 15 | 7 | 4 | 2 |
| 0.300 | 0.667 | 0.824 | 14 | 7 | 4 | 3 |
| 0.400 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.500 | 0.737 | 0.824 | 14 | 5 | 6 | 3 |
| 0.600 | 0.812 | 0.765 | 13 | 3 | 8 | 4 |
| 0.700 | 0.800 | 0.471 | 8 | 2 | 9 | 9 |
| 0.800 | 0.800 | 0.471 | 8 | 2 | 9 | 9 |
| 0.900 | 0.875 | 0.412 | 7 | 1 | 10 | 10 |

### text (n=28)
- Accuracy: 0.857 | ROC-AUC: 0.914 | Average Precision: 0.947
- Label=1 P/R/F1: 0.933 / 0.824 / 0.875
- Macro avg P/R/F1: 0.851 / 0.866 / 0.854
- Weighted avg P/R/F1: 0.869 / 0.857 / 0.859
- Latency: 0.12 ms/sample (total 0.003 s)
Plots:
![text_roc_curve.png](text_roc_curve.png)
![text_pr_curve.png](text_pr_curve.png)
![text_prob_distribution.png](text_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.607 | 1.000 | 17 | 11 | 0 | 0 |
| 0.200 | 0.773 | 1.000 | 17 | 5 | 6 | 0 |
| 0.300 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.400 | 0.938 | 0.882 | 15 | 1 | 10 | 2 |
| 0.500 | 0.933 | 0.824 | 14 | 1 | 10 | 3 |
| 0.600 | 0.929 | 0.765 | 13 | 1 | 10 | 4 |
| 0.700 | 0.929 | 0.765 | 13 | 1 | 10 | 4 |
| 0.800 | 1.000 | 0.235 | 4 | 0 | 11 | 13 |
| 0.900 | 0.000 | 0.000 | 0 | 0 | 11 | 17 |

### st (n=28)
- Accuracy: 0.857 | ROC-AUC: 0.888 | Average Precision: 0.939
- Label=1 P/R/F1: 0.933 / 0.824 / 0.875
- Macro avg P/R/F1: 0.851 / 0.866 / 0.854
- Weighted avg P/R/F1: 0.869 / 0.857 / 0.859
- Latency: 5.71 ms/sample (total 0.160 s)
Plots:
![st_roc_curve.png](st_roc_curve.png)
![st_pr_curve.png](st_pr_curve.png)
![st_prob_distribution.png](st_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.607 | 1.000 | 17 | 11 | 0 | 0 |
| 0.200 | 0.615 | 0.941 | 16 | 10 | 1 | 1 |
| 0.300 | 0.800 | 0.941 | 16 | 4 | 7 | 1 |
| 0.400 | 0.933 | 0.824 | 14 | 1 | 10 | 3 |
| 0.500 | 0.933 | 0.824 | 14 | 1 | 10 | 3 |
| 0.600 | 0.929 | 0.765 | 13 | 1 | 10 | 4 |
| 0.700 | 0.923 | 0.706 | 12 | 1 | 10 | 5 |
| 0.800 | 1.000 | 0.412 | 7 | 0 | 11 | 10 |
| 0.900 | 1.000 | 0.059 | 1 | 0 | 11 | 16 |

### ensemble_gbdt_tfidf (n=28)
- Accuracy: 0.786 | ROC-AUC: 0.840 | Average Precision: 0.897
- Label=1 P/R/F1: 0.824 / 0.824 / 0.824
- Macro avg P/R/F1: 0.775 / 0.775 / 0.775
- Weighted avg P/R/F1: 0.786 / 0.786 / 0.786
- Latency: 0.26 ms/sample (total 0.007 s)
Plots:
![ensemble_gbdt_tfidf_roc_curve.png](ensemble_gbdt_tfidf_roc_curve.png)
![ensemble_gbdt_tfidf_pr_curve.png](ensemble_gbdt_tfidf_pr_curve.png)
![ensemble_gbdt_tfidf_prob_distribution.png](ensemble_gbdt_tfidf_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.714 | 0.882 | 15 | 6 | 5 | 2 |
| 0.200 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.300 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.400 | 0.778 | 0.824 | 14 | 4 | 7 | 3 |
| 0.500 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.600 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.700 | 0.875 | 0.824 | 14 | 2 | 9 | 3 |
| 0.800 | 0.933 | 0.824 | 14 | 1 | 10 | 3 |
| 0.900 | 0.900 | 0.529 | 9 | 1 | 10 | 8 |

### ensemble_gbdt_st (n=28)
- Accuracy: 0.750 | ROC-AUC: 0.834 | Average Precision: 0.903
- Label=1 P/R/F1: 0.778 / 0.824 / 0.800
- Macro avg P/R/F1: 0.739 / 0.730 / 0.733
- Weighted avg P/R/F1: 0.747 / 0.750 / 0.748
- Latency: 2.13 ms/sample (total 0.060 s)
Plots:
![ensemble_gbdt_st_roc_curve.png](ensemble_gbdt_st_roc_curve.png)
![ensemble_gbdt_st_pr_curve.png](ensemble_gbdt_st_pr_curve.png)
![ensemble_gbdt_st_prob_distribution.png](ensemble_gbdt_st_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.200 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.300 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.400 | 0.737 | 0.824 | 14 | 5 | 6 | 3 |
| 0.500 | 0.778 | 0.824 | 14 | 4 | 7 | 3 |
| 0.600 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.700 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.800 | 0.875 | 0.824 | 14 | 2 | 9 | 3 |
| 0.900 | 0.929 | 0.765 | 13 | 1 | 10 | 4 |

### ensemble_xgb_tfidf (n=28)
- Accuracy: 0.786 | ROC-AUC: 0.813 | Average Precision: 0.879
- Label=1 P/R/F1: 0.824 / 0.824 / 0.824
- Macro avg P/R/F1: 0.775 / 0.775 / 0.775
- Weighted avg P/R/F1: 0.786 / 0.786 / 0.786
- Latency: 0.64 ms/sample (total 0.018 s)
Plots:
![ensemble_xgb_tfidf_roc_curve.png](ensemble_xgb_tfidf_roc_curve.png)
![ensemble_xgb_tfidf_pr_curve.png](ensemble_xgb_tfidf_pr_curve.png)
![ensemble_xgb_tfidf_prob_distribution.png](ensemble_xgb_tfidf_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.714 | 0.882 | 15 | 6 | 5 | 2 |
| 0.200 | 0.714 | 0.882 | 15 | 6 | 5 | 2 |
| 0.300 | 0.737 | 0.824 | 14 | 5 | 6 | 3 |
| 0.400 | 0.737 | 0.824 | 14 | 5 | 6 | 3 |
| 0.500 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.600 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.700 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.800 | 0.846 | 0.647 | 11 | 2 | 9 | 6 |
| 0.900 | 0.917 | 0.647 | 11 | 1 | 10 | 6 |

### ensemble_xgb_st (n=28)
- Accuracy: 0.714 | ROC-AUC: 0.802 | Average Precision: 0.883
- Label=1 P/R/F1: 0.737 / 0.824 / 0.778
- Macro avg P/R/F1: 0.702 / 0.684 / 0.689
- Weighted avg P/R/F1: 0.709 / 0.714 / 0.708
- Latency: 2.23 ms/sample (total 0.063 s)
Plots:
![ensemble_xgb_st_roc_curve.png](ensemble_xgb_st_roc_curve.png)
![ensemble_xgb_st_pr_curve.png](ensemble_xgb_st_pr_curve.png)
![ensemble_xgb_st_prob_distribution.png](ensemble_xgb_st_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.714 | 0.882 | 15 | 6 | 5 | 2 |
| 0.200 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.300 | 0.700 | 0.824 | 14 | 6 | 5 | 3 |
| 0.400 | 0.737 | 0.824 | 14 | 5 | 6 | 3 |
| 0.500 | 0.737 | 0.824 | 14 | 5 | 6 | 3 |
| 0.600 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.700 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.800 | 0.786 | 0.647 | 11 | 3 | 8 | 6 |
| 0.900 | 0.917 | 0.647 | 11 | 1 | 10 | 6 |

### ensemble_rf_tfidf (n=28)
- Accuracy: 0.821 | ROC-AUC: 0.856 | Average Precision: 0.903
- Label=1 P/R/F1: 0.875 / 0.824 / 0.848
- Macro avg P/R/F1: 0.812 / 0.821 / 0.816
- Weighted avg P/R/F1: 0.826 / 0.821 / 0.823
- Latency: 1.83 ms/sample (total 0.051 s)
Plots:
![ensemble_rf_tfidf_roc_curve.png](ensemble_rf_tfidf_roc_curve.png)
![ensemble_rf_tfidf_pr_curve.png](ensemble_rf_tfidf_pr_curve.png)
![ensemble_rf_tfidf_prob_distribution.png](ensemble_rf_tfidf_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.714 | 0.882 | 15 | 6 | 5 | 2 |
| 0.200 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.300 | 0.778 | 0.824 | 14 | 4 | 7 | 3 |
| 0.400 | 0.875 | 0.824 | 14 | 2 | 9 | 3 |
| 0.500 | 0.875 | 0.824 | 14 | 2 | 9 | 3 |
| 0.600 | 0.933 | 0.824 | 14 | 1 | 10 | 3 |
| 0.700 | 0.933 | 0.824 | 14 | 1 | 10 | 3 |
| 0.800 | 0.933 | 0.824 | 14 | 1 | 10 | 3 |
| 0.900 | 0.875 | 0.412 | 7 | 1 | 10 | 10 |

### ensemble_rf_st (n=28)
- Accuracy: 0.821 | ROC-AUC: 0.834 | Average Precision: 0.908
- Label=1 P/R/F1: 0.875 / 0.824 / 0.848
- Macro avg P/R/F1: 0.812 / 0.821 / 0.816
- Weighted avg P/R/F1: 0.826 / 0.821 / 0.823
- Latency: 3.58 ms/sample (total 0.100 s)
Plots:
![ensemble_rf_st_roc_curve.png](ensemble_rf_st_roc_curve.png)
![ensemble_rf_st_pr_curve.png](ensemble_rf_st_pr_curve.png)
![ensemble_rf_st_prob_distribution.png](ensemble_rf_st_prob_distribution.png)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.682 | 0.882 | 15 | 7 | 4 | 2 |
| 0.200 | 0.737 | 0.824 | 14 | 5 | 6 | 3 |
| 0.300 | 0.778 | 0.824 | 14 | 4 | 7 | 3 |
| 0.400 | 0.824 | 0.824 | 14 | 3 | 8 | 3 |
| 0.500 | 0.875 | 0.824 | 14 | 2 | 9 | 3 |
| 0.600 | 0.875 | 0.824 | 14 | 2 | 9 | 3 |
| 0.700 | 0.933 | 0.824 | 14 | 1 | 10 | 3 |
| 0.800 | 0.929 | 0.765 | 13 | 1 | 10 | 4 |
| 0.900 | 0.917 | 0.647 | 11 | 1 | 10 | 6 |

### llm (n=28)
- Accuracy: 0.750 | ROC-AUC: 0.714 | Average Precision: 0.733
- Label=1 P/R/F1: 0.750 / 0.882 / 0.811
- Macro avg P/R/F1: 0.750 / 0.714 / 0.721
- Weighted avg P/R/F1: 0.750 / 0.750 / 0.740
- Latency: 200.00 ms/sample (total 0.000 s)

| Threshold | Precision | Recall | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.200 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.300 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.400 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.500 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.600 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.700 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.800 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |
| 0.900 | 0.750 | 0.882 | 15 | 5 | 6 | 2 |

## Cost & latency comparison

| Model | Precision | Recall | Latency (ms) | Cost $/1M | Speedup vs Claude | Savings vs Claude |
| --- | --- | --- | --- | --- | --- | --- |
| claude | 1.000 | 1.000 | 1500.00 | 1800.0000 | 1.000 | 1.000 |
| gbdt | 0.700 | 0.824 | 1.43 | 0.0544 | 1049.649 | 33063.944 |
| xgb | 0.688 | 0.647 | 0.53 | 0.0202 | 2834.319 | 89281.045 |
| rf | 0.737 | 0.824 | 1.76 | 0.0672 | 850.194 | 26781.105 |
| text | 0.933 | 0.824 | 0.12 | 0.0044 | 12964.963 | 408396.344 |
| st | 0.933 | 0.824 | 5.71 | 2.2810 | 262.518 | 789.120 |
| ensemble_gbdt_tfidf | 0.824 | 0.824 | 0.26 | 0.0101 | 5674.654 | 178751.585 |
| ensemble_gbdt_st | 0.778 | 0.824 | 2.13 | 0.8522 | 702.653 | 2112.148 |
| ensemble_xgb_tfidf | 0.824 | 0.824 | 0.64 | 0.0244 | 2339.187 | 73684.379 |
| ensemble_xgb_st | 0.737 | 0.824 | 2.23 | 0.8918 | 671.495 | 2018.490 |
| ensemble_rf_tfidf | 0.875 | 0.824 | 1.83 | 0.0696 | 820.702 | 25852.109 |
| ensemble_rf_st | 0.875 | 0.824 | 3.58 | 1.4272 | 419.564 | 1261.193 |
| llm | 0.750 | 0.882 | 200.00 | 79.8413 | 7.500 | 22.545 |