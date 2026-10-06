# Polynomial Regression Assignment - BT2024121

This repository contains the training, model-selection and inference code for the two personalized polynomial-regression problems.

## Final models

- **var1:** degree 5 polynomial features + StandardScaler + Lasso(alpha=0.008)
- **var2:** degree 10 polynomial features + StandardScaler + Ridge(alpha=0.5)

Both are polynomial regression models. Regularization is used only on the linear coefficients after polynomial feature expansion to reduce overfitting.

## Expected data files

Create a `data/` directory and place:

- `BT2024121_train_var1.csv`
- `BT2024121_test_var1.csv`
- `BT2024121_train_var2.csv`
- `BT2024121_test_var2.csv`

The personalized datasets are intentionally not included in the repository bundle.

## Run

```bash
pip install -r requirements.txt
python model_selection.py
python train_and_predict.py --data-dir data --output-dir outputs
```

The prediction files will be written to `outputs/` with a single column named `y`.

## Report

Before submission, add the actual GitHub repository URL to the provided PDF report.
