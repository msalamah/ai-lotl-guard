from lotl_detector.models.calibrate import calibrate_threshold


def test_calibrate_threshold_hits_target_recall():
    probabilities = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    result = calibrate_threshold(probabilities, labels, recall_target=0.9)
    assert result.achieved_target is True
    assert result.recall >= 0.9
    assert result.threshold <= 0.8
