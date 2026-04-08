import argparse
import os
import pandas as pd
from minio import Minio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0)
parser.add_argument("--max_features", type=int, default=10000)
args = parser.parse_args()

print("Telechargement du CSV depuis MinIO...")
os.makedirs("/tmp/data", exist_ok=True)

client = Minio(
    "minio-service.kubeflow:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)
client.fget_object(
    "mlpipeline",
    "private-artifacts/azemo/v2/artifacts/amazon-reviews-pipeline/ae2c172a-3f7a-4f11-94a1-945f2c18e926/preprocess-data/2c637baf-f05a-4e3b-bf5a-d9bda7410c52/output_dataset",
    "/tmp/data/amazon_data.csv"
)
print("CSV telecharge !")

df = pd.read_csv("/tmp/data/amazon_data.csv")
df = df.dropna(subset=["reviewText", "label"])
print(f"Shape: {df.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    df["reviewText"], df["label"], test_size=0.2, random_state=42
)
tfidf = TfidfVectorizer(max_features=args.max_features)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
model = LogisticRegression(C=args.C, max_iter=1000)
model.fit(X_train_vec, y_train)
acc = accuracy_score(y_test, model.predict(X_test_vec))
print(f"accuracy={acc:.4f}")
