"""
Train and save the Smart Waste Management overflow-prediction model.
Run this to (re)generate smart_waste_model.pkl and label_encoders.pkl
for the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SMART WASTE MANAGEMENT - MODEL TRAINING")
print("=" * 60)

# 1. Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('dataset.csv')
print(f"   Dataset shape: {df.shape}")

# 2. Missing values
print("\n2. Checking missing values...")
print(f"   Missing values: {df.isnull().sum().sum()}")

# 3. Duplicates
print("\n3. Removing duplicates...")
before = df.shape[0]
df = df.drop_duplicates()
print(f"   Duplicates removed: {before - df.shape[0]}")

# 4. Encode categorical columns (store the mapping so the app can use it)
print("\n4. Encoding categorical columns...")
categorical_cols = ['Sensor_Status', 'Motion_Detected', 'Rain_Detected', 'Overflow']
encoders = {}
for col in categorical_cols:
    df[col] = df[col].astype('category')
    encoders[col] = dict(enumerate(df[col].cat.categories))  # code -> label
    df[col] = df[col].cat.codes
print(f"   Encoded columns: {categorical_cols}")

# 5. Features / target (11 features, same as the notebook)
feature_cols = ['Current_Fill_%', 'Weight_kg', 'Temperature_C', 'Humidity_%',
                 'Methane_ppm', 'CO2_ppm', 'Battery_Level_%',
                 'Hours_Since_Last_Collection', 'Sensor_Status',
                 'Motion_Detected', 'Rain_Detected']

X = df[feature_cols]
y = df['Overflow']
print(f"\n5. Features shape: {X.shape}, Target shape: {y.shape}")

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n6. Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# 7. Train
print("\n7. Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("   Done.")

# 8. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n8. Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=['No Overflow', 'Overflow']))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 9. Feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n9. Feature Importance:")
print(importance_df.to_string(index=False))

# 10. Save model + encoders + metadata used by the app
print("\n10. Saving artifacts...")
with open('smart_waste_model.pkl', 'wb') as f:
    pickle.dump(model, f)

metadata = {
    'feature_cols': feature_cols,
    'encoders': encoders,           # e.g. {'Sensor_Status': {0:'Fault',1:'Healthy',2:'Warning'}, ...}
    'accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'feature_importance': importance_df.to_dict(orient='records'),
    'n_train': int(X_train.shape[0]),
    'n_test': int(X_test.shape[0]),
}
with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("    Saved smart_waste_model.pkl and model_metadata.pkl")
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE — run: streamlit run app.py")
print("=" * 60)
