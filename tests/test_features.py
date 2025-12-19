import pandas as pd

from lotl_detector.features import extract_features, build_feature_frame


def test_keyword_flags_and_metrics():
    row = pd.Series(
        {
            "CommandLine": 'powershell.exe -enc aGVsbG8= -Command "Invoke-WebRequest http://example.com"',
            "SourceImage": r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
        }
    )
    feats = extract_features(row)
    assert feats["has_powershell"] == 1.0
    assert feats["has_encodedcommand"] == 1.0
    assert feats["has_download"] == 1.0
    assert feats["cmd_token_count"] >= 4
    assert feats["source_image_base"] == "powershell.exe"
    assert feats["cmd_exe_base"] == "powershell.exe"


def test_feature_frame_shapes():
    df = pd.DataFrame(
        [
            {"CommandLine": "cmd.exe /c whoami", "SourceImage": r"C:\\Windows\\System32\\cmd.exe"},
            {"CommandLine": "bitsadmin /transfer job1", "SourceImage": r"C:\\Windows\\System32\\bitsadmin.exe"},
        ]
    )
    feature_df = build_feature_frame(df)
    assert feature_df.shape[0] == 2
    assert "has_bitsadmin" in feature_df.columns
    assert feature_df.loc[1, "has_bitsadmin"] == 1.0
