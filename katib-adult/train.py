import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_split", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age","workclass","fnlwgt","education","education-num",
               "marital-status","occupation","relationship","race","sex",
               "capital-gain","capital-loss","hours-per-week","native-country","income"]
    df = pd.read_csv(url, header=None, names=columns, na_values=" ?")
    df.dropna(inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.drop("income", axis=1)
    y = df["income"]
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    )
    clf.fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    print(f"accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
