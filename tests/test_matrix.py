import pandas as pd

from lotl_detector.models.matrix import prepare_feature_matrix


def test_prepare_feature_matrix_supplies_missing_columns_and_types():
    df = pd.DataFrame(
        [
            {
                "CommandLine": "powershell.exe -enc ABCDEFG",
                "SourceImage": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            }
        ]
    )
    feature_list = ["cmd_length", "source_image_base", "cmd_exe_base", "custom_numeric"]
    cat_features = ["source_image_base", "cmd_exe_base"]

    matrix = prepare_feature_matrix(df, feature_list, cat_features)

    assert list(matrix.columns) == feature_list
    assert matrix["custom_numeric"].iloc[0] == 0.0
    assert str(matrix["source_image_base"].dtype) == "category"
