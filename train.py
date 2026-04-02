import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import boto3

# 1. Baca data dari S3 (Menggunakan pandas + s3fs)
df = pd.read_csv("s3://lks-2026-processed-02/iris_processed.csv")

# 2. Training (Asumsi kolom target adalah 'Species')
X = df.iloc[:, 1:-1] # Fitur
y = df.iloc[:, -1]   # Label/Target

model = RandomForestClassifier()
model.fit(X, y)

# 3. Simpan dan Upload ke S3 Models
joblib.dump(model, "model.tar.gz")

s3 = boto3.client('s3')
s3.upload_file("model.tar.gz", "lks-2026-models-02", "iris_model.tar.gz")
print("Sukses! Model telah di-upload ke S3 Models.")
