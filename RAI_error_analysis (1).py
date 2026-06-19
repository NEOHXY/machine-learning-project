import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from responsibleai import RAIInsights
from raiwidgets import ResponsibleAIDashboard
import time

# 1. Load Data
print(">>> [Step 1] Loading Data...")
try:
    train_df = pd.read_csv('Sensitive 0.1 codinate from nose train.csv')
    test_df = pd.read_csv('Sensitive 0.1 codinate from nose test.csv')
except Exception as e:
    print(f"Error: {e}")
    exit()

target_feature = train_df.columns[-1]


# 2. Data Cleaning
print(">>> [Step 2] Cleaning Data...")
# Fill missing values to ensure stability
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
train_means = train_df[numeric_cols].mean()
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_means)
test_df[numeric_cols] = test_df[numeric_cols].fillna(train_means)

X_train = train_df.drop(columns=[target_feature])
y_train = train_df[target_feature]


# 3. Train Surrogate Model
print(">>> [Step 3] Training Surrogate Model...")
model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# 4. Configure RAI Components
print(">>> [Step 4] Configuring Dashboard Components...")

rai_insights = RAIInsights(
    model=model,
    train=train_df,
    test=test_df,
    target_column=target_feature,
    task_type="classification"
)

# Error Analysis
rai_insights.error_analysis.add()

# InterpretML 
rai_insights.explainer.add()

# Data Balance
print(">>> [Step 5] Computing Insights...")
rai_insights.compute()


# 5. Launch Dashboard
PORT = 6001
print("="*60)
print(f"RAI DASHBOARD STARTED")
print(f"URL: http://localhost:{PORT}")
print("="*60)

dashboard = ResponsibleAIDashboard(rai_insights, port=PORT)

while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        break