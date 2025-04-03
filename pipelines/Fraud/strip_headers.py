# to remove headers form test.csv as FraudTransform can not accept file with headers and for for test with headers will be needed for evalute.py
import pandas as pd
import os

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    input_path = f"{base_dir}/input/test.csv"
    output_path = f"{base_dir}/output/test_no_headers.csv"
    df = pd.read_csv(input_path)
    df.to_csv(output_path, header=False, index=False)