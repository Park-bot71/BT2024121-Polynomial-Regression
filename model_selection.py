import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLLNO = "BT2024121"

def load_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    names = arr.dtype.names
    feature_names = [n for n in names if n != "y"]
    X = np.column_stack([arr[n] for n in feature_names])
    y = arr["y"]
    return X, y

def evaluate(X, y, degrees, estimator, alpha):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        XP = poly.fit_transform(X)
        mses, r2s = [], []

        for train_idx, val_idx in kf.split(XP):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(XP[train_idx])
            X_val = scaler.transform(XP[val_idx])

            if estimator == "lasso":
                model = Lasso(alpha=alpha, max_iter=50000)
            else:
                model = Ridge(alpha=alpha)

            model.fit(X_train, y[train_idx])
            pred = model.predict(X_val)
            mses.append(mean_squared_error(y[val_idx], pred))
            r2s.append(r2_score(y[val_idx], pred))

        print(
            f"degree={degree:2d} features={XP.shape[1]:4d} "
            f"MSE={np.mean(mses):.6f} R2={np.mean(r2s):.6f}"
        )

if __name__ == "__main__":
    X1, y1 = load_csv(f"data/{ROLLNO}_train_var1.csv")
    X2, y2 = load_csv(f"data/{ROLLNO}_train_var2.csv")

    print("VAR1 - Lasso polynomial regression (alpha=0.008)")
    evaluate(X1, y1, range(1, 8), "lasso", 0.008)

    print("\nVAR2 - Ridge polynomial regression (alpha=0.5)")
    evaluate(X2, y2, range(4, 15), "ridge", 0.5)
