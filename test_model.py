import joblib
import pandas as pd

# Load model yang baru dibuat
model = joblib.load("model.tar.gz")

# Data uji (SepalLength, SepalWidth, PetalLength, PetalWidth)
# Contoh data Iris Virginica/Versicolor
sample_data = [[5.1, 3.5, 1.4, 0.2]] 
prediction = model.predict(sample_data)

print(f"Hasil Prediksi untuk data {sample_data}: {prediction[0]}")
