import pandas as pd
import boto3

raw_path = "s3://lks-2026-raw-02/raw/dataset.csv"
processed_path = "s3://lks-2026-processed-02/iris_processed.csv"

print("Sedang memproses data...")
df = pd.read_csv(raw_path)
df.to_csv(processed_path, index=False)
print(f"Sukses! Data tersimpan di {processed_path}")
