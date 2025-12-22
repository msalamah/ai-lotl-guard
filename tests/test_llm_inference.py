from lotl_detector.models.llm_inference import LocalLLMReasoner


def test_parse_output_handles_json():
    text = '{"label": "malicious", "explanation": "Executes LOLBin."}'
    parsed = LocalLLMReasoner._parse_output(text)
    assert parsed["label"] == "malicious"
    assert parsed["explanation"] == "Executes LOLBin."


def test_parse_output_handles_text():
    text = "Likely benign due to normal parent process."
    parsed = LocalLLMReasoner._parse_output(text)
    assert parsed["label"] == "unknown"
    assert parsed["explanation"].startswith("Likely benign")
