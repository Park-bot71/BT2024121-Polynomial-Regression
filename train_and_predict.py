import argparse
import csv
import os
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline

ROLLNO = "BT2024121"

def load_csv(path, target=True):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    names = arr.dtype.names
    features = [name for name in names if name != "y"]
    X = np.column_stack([arr[name] for name in features])
    y = arr["y"] if target and "y" in names else None
    return X, y

def write_predictions(path, predictions):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y"])
        for value in predictions:
            writer.writerow([f"{float(value):.15g}"])

def main(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X1, y1 = load_csv(os.path.join(data_dir, f"{ROLLNO}_train_var1.csv"))
    T1, _ = load_csv(os.path.join(data_dir, f"{ROLLNO}_test_var1.csv"), target=False)

    X2, y2 = load_csv(os.path.join(data_dir, f"{ROLLNO}_train_var2.csv"))
    T2, _ = load_csv(os.path.join(data_dir, f"{ROLLNO}_test_var2.csv"), target=False)

    model1 = Pipeline([
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("scale", StandardScaler()),
        ("reg", Lasso(alpha=0.008, max_iter=50000)),
    ])

    model2 = Pipeline([
        ("poly", PolynomialFeatures(degree=10, include_bias=False)),
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=0.5)),
    ])

    model1.fit(X1, y1)
    model2.fit(X2, y2)

    pred1 = model1.predict(T1)
    pred2 = model2.predict(T2)

    write_predictions(os.path.join(output_dir, f"{ROLLNO}_pred_var1.csv"), pred1)
    write_predictions(os.path.join(output_dir, f"{ROLLNO}_pred_var2.csv"), pred2)

    print("Created both prediction files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
