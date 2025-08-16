import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt

# สร้างข้อมูล training สำหรับตัวอย่าง
# ใช้ฟังก์ชัน y = x^2 + noise เป็นตัวอย่าง
np.random.seed(42)
X_train = np.random.uniform(1, 10, 100).reshape(-1, 1)
y_train = (X_train.flatten() ** 2) + np.random.normal(0, 5, 100)

print("=== CatBoost Regression Example ===")
print("Training data shape:", X_train.shape)
print("First 5 training samples:")
for i in range(5):
    print(f"X: {X_train[i][0]:.2f}, y: {y_train[i]:.2f}")

# สร้างและ train CatBoost Regressor
model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=False  # ปิดการแสดงผลระหว่าง training
)

# Training model
model.fit(X_train, y_train)

# ข้อมูลที่ต้องการทำนาย
X_predict = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# ทำนายผลลัพธ์
predictions = model.predict(X_predict)

print("\n=== Prediction Results ===")
print("Input -> Prediction")
for i, (x, pred) in enumerate(zip(X_predict.flatten(), predictions)):
    print(f"x={x:2d} -> y={pred:7.2f}")

# คำนวณ feature importance
feature_importance = model.get_feature_importance()
print(f"\nFeature Importance: {feature_importance[0]:.4f}")

# สร้างกราฟเปรียบเทียบ
plt.figure(figsize=(10, 6))
plt.scatter(X_train.flatten(), y_train, alpha=0.6, label='Training Data', color='lightblue')
plt.plot(X_predict.flatten(), predictions, 'ro-', label='Predictions', markersize=8)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('CatBoost Regression: Training Data vs Predictions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "="*50)
print("=== CatBoost Classification Example ===")

# ตัวอย่าง Classification
# สร้างข้อมูล binary classification
X_train_cls = np.random.uniform(1, 10, 200).reshape(-1, 1)
y_train_cls = (X_train_cls.flatten() > 5).astype(int)  # 0 if x <= 5, 1 if x > 5

# สร้าง CatBoost Classifier
classifier = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    verbose=False
)

# Training classifier
classifier.fit(X_train_cls, y_train_cls)

# ทำนาย class และ probability
class_predictions = classifier.predict(X_predict)
probabilities = classifier.predict_proba(X_predict)

print("Classification Results:")
print("Input -> Class (Prob_0, Prob_1)")
for i, (x, cls, prob) in enumerate(zip(X_predict.flatten(), class_predictions, probabilities)):
    print(f"x={x:2d} -> Class {cls} ({prob[0]:.3f}, {prob[1]:.3f})")

# Model evaluation metrics
from sklearn.metrics import accuracy_score, classification_report

# ทดสอบกับข้อมูล training (ควรใช้ validation set จริงๆ)
train_pred = classifier.predict(X_train_cls)
accuracy = accuracy_score(y_train_cls, train_pred)
print(f"\nTraining Accuracy: {accuracy:.4f}")

print("\n=== Tips for Using CatBoost ===")
print("1. CatBoost จัดการ categorical features ได้อัตโนมัติ")
print("2. ไม่ต้อง scale features")
print("3. มี built-in overfitting protection")
print("4. สามารถใช้ early_stopping_rounds ได้")
print("5. รองรับ GPU training ด้วย task_type='GPU'")