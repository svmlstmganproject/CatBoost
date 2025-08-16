import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report

# สร้างข้อมูล training สำหรับตัวอย่าง
np.random.seed(42)
X_train = np.random.uniform(1, 10, 100).reshape(-1, 1)
y_train = (X_train.flatten() ** 2) + np.random.normal(0, 5, 100)

print("=== CatBoost vs XGBoost Regression Comparison ===")
print("Training data shape:", X_train.shape)

# ข้อมูลที่ต้องการทำนาย
X_predict = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# === CatBoost Regressor ===
catboost_model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=False
)
catboost_model.fit(X_train, y_train)
catboost_pred = catboost_model.predict(X_predict)
catboost_train_pred = catboost_model.predict(X_train)

# === XGBoost Regressor ===
xgboost_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=0
)
xgboost_model.fit(X_train, y_train)
xgboost_pred = xgboost_model.predict(X_predict)
xgboost_train_pred = xgboost_model.predict(X_train)

# คำนวณ metrics สำหรับ training data
catboost_mse = mean_squared_error(y_train, catboost_train_pred)
catboost_mae = mean_absolute_error(y_train, catboost_train_pred)

xgboost_mse = mean_squared_error(y_train, xgboost_train_pred)
xgboost_mae = mean_absolute_error(y_train, xgboost_train_pred)

print("\n=== Regression Performance Metrics ===")
print(f"CatBoost - MSE: {catboost_mse:.4f}, MAE: {catboost_mae:.4f}")
print(f"XGBoost  - MSE: {xgboost_mse:.4f}, MAE: {xgboost_mae:.4f}")

# แสดงผลการทำนาย
print("\n=== Prediction Comparison ===")
print("Input | CatBoost | XGBoost  | Difference")
print("-" * 45)
for i, (x, cat_pred, xgb_pred) in enumerate(zip(X_predict.flatten(), catboost_pred, xgboost_pred)):
    diff = abs(cat_pred - xgb_pred)
    print(f"{x:5d} | {cat_pred:8.2f} | {xgb_pred:8.2f} | {diff:8.2f}")

# สร้างกราฟเปรียบเทียบ Regression
plt.figure(figsize=(15, 10))

# กราฟที่ 1: Training Data และ Predictions
plt.subplot(2, 2, 1)
plt.scatter(X_train.flatten(), y_train, alpha=0.6, label='Training Data', color='lightgray', s=30)
plt.plot(X_predict.flatten(), catboost_pred, 'ro-', label='CatBoost', markersize=8, linewidth=2)
plt.plot(X_predict.flatten(), xgboost_pred, 'bs-', label='XGBoost', markersize=8, linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression: CatBoost vs XGBoost Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# กราฟที่ 2: ความแตกต่างของการทำนาย
plt.subplot(2, 2, 2)
differences = np.abs(catboost_pred - xgboost_pred)
plt.plot(X_predict.flatten(), differences, 'go-', markersize=8, linewidth=2)
plt.xlabel('X')
plt.ylabel('Absolute Difference')
plt.title('Absolute Difference between Models')
plt.grid(True, alpha=0.3)

# === Classification Comparison ===
print("\n" + "="*60)
print("=== CatBoost vs XGBoost Classification Comparison ===")

# สร้างข้อมูล classification
X_train_cls = np.random.uniform(1, 10, 200).reshape(-1, 1)
y_train_cls = (X_train_cls.flatten() > 5).astype(int)

# CatBoost Classifier
catboost_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    verbose=False
)
catboost_clf.fit(X_train_cls, y_train_cls)
catboost_cls_pred = catboost_clf.predict(X_predict)
catboost_cls_prob = catboost_clf.predict_proba(X_predict)
catboost_train_cls_pred = catboost_clf.predict(X_train_cls)

# XGBoost Classifier
xgboost_clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    verbosity=0
)
xgboost_clf.fit(X_train_cls, y_train_cls)
xgboost_cls_pred = xgboost_clf.predict(X_predict)
xgboost_cls_prob = xgboost_clf.predict_proba(X_predict)
xgboost_train_cls_pred = xgboost_clf.predict(X_train_cls)

# คำนวณ accuracy
catboost_accuracy = accuracy_score(y_train_cls, catboost_train_cls_pred)
xgboost_accuracy = accuracy_score(y_train_cls, xgboost_train_cls_pred)

print(f"\n=== Classification Performance ===")
print(f"CatBoost Accuracy: {catboost_accuracy:.4f}")
print(f"XGBoost Accuracy:  {xgboost_accuracy:.4f}")

print("\n=== Classification Predictions ===")
print("Input | CatBoost | XGBoost | CB_Prob_1 | XGB_Prob_1")
print("-" * 55)
for i, (x, cat_cls, xgb_cls, cat_prob, xgb_prob) in enumerate(
    zip(X_predict.flatten(), catboost_cls_pred, xgboost_cls_pred, 
        catboost_cls_prob, xgboost_cls_prob)):
    print(f"{x:5d} | {cat_cls:8d} | {xgb_cls:7d} | {cat_prob[1]:9.3f} | {xgb_prob[1]:8.3f}")

# กราฟที่ 3: Classification Results
plt.subplot(2, 2, 3)
x_vals = X_predict.flatten()
width = 0.35
x_pos = np.arange(len(x_vals))

plt.bar(x_pos - width/2, catboost_cls_pred, width, label='CatBoost', alpha=0.8)
plt.bar(x_pos + width/2, xgboost_cls_pred, width, label='XGBoost', alpha=0.8)
plt.xlabel('Input Value')
plt.ylabel('Predicted Class')
plt.title('Classification: CatBoost vs XGBoost')
plt.xticks(x_pos, x_vals)
plt.legend()
plt.grid(True, alpha=0.3)

# กราฟที่ 4: Probability Comparison
plt.subplot(2, 2, 4)
plt.plot(x_vals, catboost_cls_prob[:, 1], 'ro-', label='CatBoost Prob(Class=1)', markersize=8)
plt.plot(x_vals, xgboost_cls_prob[:, 1], 'bs-', label='XGBoost Prob(Class=1)', markersize=8)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Decision Boundary')
plt.xlabel('Input Value')
plt.ylabel('Probability of Class 1')
plt.title('Classification Probabilities Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Model Summary ===
print("\n" + "="*60)
print("=== FINAL COMPARISON SUMMARY ===")
print(f"Regression:")
print(f"  CatBoost MSE: {catboost_mse:.4f}")
print(f"  XGBoost MSE:  {xgboost_mse:.4f}")
print(f"  Winner: {'CatBoost' if catboost_mse < xgboost_mse else 'XGBoost'} (Lower MSE)")

print(f"\nClassification:")
print(f"  CatBoost Accuracy: {catboost_accuracy:.4f}")
print(f"  XGBoost Accuracy:  {xgboost_accuracy:.4f}")
print(f"  Winner: {'CatBoost' if catboost_accuracy > xgboost_accuracy else 'XGBoost'} (Higher Accuracy)")

print(f"\nAverage Prediction Difference: {np.mean(differences):.4f}")

print("\n=== Installation Requirements ===")
print("pip install catboost xgboost matplotlib scikit-learn")

print("\n=== Key Observations ===")
print("1. Both models should perform similarly on this simple dataset")
print("2. Differences may be more apparent with categorical features")
print("3. CatBoost typically requires less hyperparameter tuning")
print("4. XGBoost has more extensive ecosystem and community")
print("5. Performance depends heavily on data characteristics")