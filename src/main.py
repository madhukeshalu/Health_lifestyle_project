# ============================================================
# main.py
# HEALTH & LIFESTYLE RISK PREDICTION – END-TO-END PIPELINE
# ============================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from scipy.stats import ttest_rel

sns.set_style("whitegrid")

# ============================================================
# PATH CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "health_lifestyle_risk_raw_with_issues_10000.csv")
CLEANED_DATA = os.path.join(PROJECT_ROOT, "data", "cleaned")
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")

os.makedirs(CLEANED_DATA, exist_ok=True)
os.makedirs(PROCESSED_DATA, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

# ============================================================
# 1️⃣ DATA CLEANING
# ============================================================

print("\n🚀 STEP 1: DATA CLEANING")

df = pd.read_csv(RAW_DATA)

df.columns = (
    df.columns.str.strip().str.lower().str.replace(" ", "_")
)

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())


categorical_cols = [
    'gender','diet_quality','stress_level','smoking_status',
    'alcohol_consumption','mental_wellbeing_score',
    'chronic_conditions','family_history'
]

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

num_cols = [
    'age','height_cm','weight_kg',
    'physical_activity_hours_per_week',
    'sleep_hours_per_day',
    'mental_wellbeing_score',
    'daily_screen_time_hours',
    'water_intake_liters',
    'fast_food_frequency_per_week'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())


df = df.drop_duplicates(subset='person_id')

df['gender'] = df['gender'].map({'male': 1, 'female': 0})
df['diet_quality'] = df['diet_quality'].map({'poor':1,'average':2,'good':3})
df['stress_level'] = df['stress_level'].map({'low':1,'medium':2,'high':3})
df['smoking_status'] = df['smoking_status'].map({'non-smoker':0,'occasional':1,'regular':2})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'moderate':1,'high':2})
df['family_history'] = df['family_history'].map({'yes':1,'no':0})
df['chronic_conditions'] = df['chronic_conditions'].apply(lambda x: 0 if x == 'none' else 1)

df = df[df['age'].between(5, 100)]
df = df[df['sleep_hours_per_day'].between(3, 14)]

clean_path = os.path.join(CLEANED_DATA, "health_lifestyle_cleaned.csv")
df.to_csv(clean_path, index=False)

print("✅ Data cleaning completed")

# ============================================================
# 2️⃣ FEATURE ENGINEERING
# ============================================================

print("\n🚀 STEP 2: FEATURE ENGINEERING")

df['bmi'] = df['weight_kg'] / (df['height_cm'] ** 2)

df['wellness_score'] = (
    df['physical_activity_hours_per_week'] +
    df['sleep_hours_per_day'] +
    df['mental_wellbeing_score'] +
    (df['diet_quality'] * 2) -
    (df['stress_level'] * 2)
) / 5

def risk_category(row):
    if row['wellness_score'] < 4:
        return 'High Risk'
    elif row['wellness_score'] < 6:
        return 'Medium Risk'
    return 'Low Risk'

df['risk_category'] = df.apply(risk_category, axis=1)

processed_path = os.path.join(PROCESSED_DATA, "health_lifestyle_feature_engineered.csv")
df.to_csv(processed_path, index=False)

print("✅ Feature engineering completed")

# ============================================================
# 3️⃣ MODEL TRAINING
# ============================================================

print("\n🚀 STEP 3: MODEL TRAINING")

df = pd.read_csv(processed_path)

target_encoder = LabelEncoder()
df['risk_category_encoded'] = target_encoder.fit_transform(df['risk_category'])
joblib.dump(target_encoder, os.path.join(MODELS_DIR, "target_encoder.pkl"))

DROP_COLS = [
    "person_id","risk_category","risk_category_encoded",
    "wellness_score","bmi"
]

X = df.drop(columns=DROP_COLS)
y = df['risk_category_encoded']

joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, "feature_columns.pkl"))

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, os.path.join(MODELS_DIR, "health_risk_model.pkl"))

print("✅ Model trained & saved")

# ============================================================
# 4️⃣ MODEL EVALUATION
# ============================================================

print("\n🚀 STEP 4: MODEL EVALUATION")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred, average="weighted"))
print("AUC:", roc_auc_score(y_test, y_prob, multi_class="ovr"))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(VISUALS_DIR, "confusion_matrix.png"))
plt.close()

# ============================================================
# 5️⃣ BUSINESS ROI ANALYSIS
# ============================================================

print("\n🚀 STEP 5: BUSINESS ROI ANALYSIS")

y_true = (y_test == 2).astype(int)
y_scores = y_prob[:, 2]

def business_roi(tp, fp, fn, tn):
    return (tp * 10000) + (tn * 1000) - (fp * 5000) - (fn * 50000)

results = []

for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    roi = business_roi(tp, fp, fn, tn)
    results.append([threshold, roi])

roi_df = pd.DataFrame(results, columns=["Threshold", "ROI"])

plt.plot(roi_df["Threshold"], roi_df["ROI"], marker="o")
plt.xlabel("Threshold")
plt.ylabel("ROI")
plt.title("ROI Optimization Curve")
plt.savefig(os.path.join(VISUALS_DIR, "roi_curve.png"))
plt.close()

best = roi_df.loc[roi_df["ROI"].idxmax()]

print("\n🎯 FINAL DECISION")
print("Best Threshold:", best["Threshold"])
print("Max ROI:", best["ROI"])

print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")
