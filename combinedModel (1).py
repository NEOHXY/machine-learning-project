import os
import pandas as pd
import numpy as np
import pickle
import time
import warnings
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from responsibleai import RAIInsights
from raiwidgets import ResponsibleAIDashboard

# ==========================================
# 0. Configuration & Setup
# ==========================================
warnings.filterwarnings("ignore")

# UPDATE THIS PATH IF NEEDED
BASE_DIR = r"C:\Users\Asus\Downloads\ML project\SKLEARN\SKLEARN"

NEW_DATA_FILE = os.path.join(BASE_DIR, "confirmed_samples.csv")
ARCHIVE_FILE = os.path.join(BASE_DIR, "archived_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models_history")

TARGET_CLASSES = ['air', 'demam', 'dengar', 'makan', 'minum', 'salah', 'saya', 'senyap', 'tidur', 'waktu']
ABSOLUTE_MIN_LIMIT = 30
PORT_RAI = 6001

# ==========================================
# 1. Helper Functions (From Your Training Script)
# ==========================================
def load_and_clean_data(filepath):
    """读取并清理数据，确保格式正确"""
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return None
        
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]
        
        if 'label' not in df.columns:
            return None
            
        df['label'] = df['label'].astype(str).str.lower()
        
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
            
        return df
    except Exception as e:
        print(f"⚠️ 读取 {os.path.basename(filepath)} 失败: {e}")
        return None

def get_max_balanced_dataset(df_total):
    """平衡数据策略"""
    print(f"\n[Data Balancing] Calculating balance...")
    class_counts = df_total['label'].value_counts().to_dict()
    
    for cls in TARGET_CLASSES:
        if cls not in class_counts: class_counts[cls] = 0
            
    valid_counts = {k: v for k, v in class_counts.items() if k in TARGET_CLASSES}
    if not valid_counts: return None, "No valid classes"

    min_class_name = min(valid_counts, key=valid_counts.get)
    limit_count = valid_counts[min_class_name]
    
    print(f"📊 Class Distribution: {valid_counts}")
    print(f"📉 Limiting Factor: '{min_class_name}' (Count: {limit_count})")
    
    if limit_count < ABSOLUTE_MIN_LIMIT:
        print(f"❌ Training Cancelled: '{min_class_name}' has insufficient data ({limit_count} < {ABSOLUTE_MIN_LIMIT}).")
        return None, min_class_name

    balanced_frames = []
    for cls in TARGET_CLASSES:
        df_cls = df_total[df_total['label'] == cls]
        if len(df_cls) > limit_count:
            df_sampled = df_cls.sample(n=limit_count, random_state=42)
        else:
            df_sampled = df_cls
        balanced_frames.append(df_sampled)
    
    return pd.concat(balanced_frames), None

# ==========================================
# 2. Main Logic: Train -> Track -> Dashboard
# ==========================================
def main():
    print("="*60)
    print(">>> [Phase 1] Data Management & Training")
    print("="*60)

    if not os.path.exists(BASE_DIR):
        print(f"Error: Cannot find directory {BASE_DIR}")
        return

    # --- Step A: Load Data (New + Archive) ---
    df_new = load_and_clean_data(NEW_DATA_FILE)
    df_old = load_and_clean_data(ARCHIVE_FILE)
    
    if df_new is None and df_old is None:
        print("❌ No data found in confirmed_samples.csv OR archived_data.csv")
        return

    if df_new is not None:
        print(f"✅ Found NEW data ({len(df_new)} rows). Merging...")
        df_total = pd.concat([df_old, df_new], ignore_index=True) if df_old is not None else df_new
    else:
        print("ℹ️ No new data. Using existing archived data.")
        df_total = df_old

    # --- Step B: Balance Data ---
    df_balanced, error = get_max_balanced_dataset(df_total)
    if df_balanced is None:
        return

    # --- Step C: Prepare Train/Test Split for RAI ---
    # RAI needs a 'Test Set' to show you where errors happen.
    # We split the balanced data 80% Train, 20% Test.
    target_feature = 'label'
    X = df_balanced.drop(columns=[target_feature])
    y = df_balanced[target_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Reconstruct DataFrames for RAI (RAI needs X and y together in the dataframe)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"\n✅ Data Split: Train ({len(train_df)} rows), Test ({len(test_df)} rows)")

    # --- Step D: Train Model (Random Forest) ---
    print("\n>>> [Phase 2] Training & Tracking with MLflow")
    
    mlflow.set_experiment("Pose_Classification_Tracker")
    
    with mlflow.start_run(run_name="Integrated_RF_Training"):
        
        # 1. Train
        n_estimators = 100
        print("👉 Training Random Forest...")
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # 2. Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"✅ Model Accuracy: {acc:.4f}")

        # 3. Log to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("data_count", len(df_balanced))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # 4. Save locally (History)
        if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path_rf = os.path.join(MODEL_DIR, f"MODEL_RF_N{len(df_balanced)}_{timestamp}.pkl")
        with open(path_rf, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 Model saved to: {path_rf}")

    # --- Step E: Archive New Data ---
    # If we successfully trained, we move confirmed_samples to archive
    if df_new is not None:
        try:
            header_needed = not os.path.exists(ARCHIVE_FILE)
            # Reload raw file to preserve format
            raw_new = pd.read_csv(NEW_DATA_FILE)
            raw_new.to_csv(ARCHIVE_FILE, mode='a', header=header_needed, index=False)
            open(NEW_DATA_FILE, 'w').close() # Clear file
            print("📦 New data archived and temp file cleared.")
        except Exception as e:
            print(f"⚠️ Warning: Could not archive data: {e}")

    # ==========================================
    # 3. RAI Dashboard
    # ==========================================
    print("\n" + "="*60)
    print(">>> [Phase 3] Configuring Responsible AI Dashboard")
    print("="*60)

    # 1. Initialize Insights
    # Note: We pass the model we just trained, and the split dataframes
    rai_insights = RAIInsights(
        model=model,
        train=train_df,
        test=test_df,
        target_column=target_feature,
        task_type="classification"
    )

    # 2. Add Components
    print("... Adding Error Analysis")
    rai_insights.error_analysis.add()
    
    print("... Adding Explainer")
    rai_insights.explainer.add()

    # 3. Compute
    print("... Computing Insights (This may take a moment)")
    rai_insights.compute()

    # 4. Launch
    print("\n" + "="*60)
    print(f"🚀 DASHBOARD READY")
    print("------------------------------------------------")
    print(f"1. [RAI Dashboard] Open: http://localhost:{PORT_RAI}")
    print(f"   (See Error Analysis Tree & Feature Importance)")
    print("------------------------------------------------")
    print(f"2. [MLflow UI] Run in terminal: 'mlflow ui'")
    print(f"   (See Model History & Metrics)")
    print("="*60)

    # Keep alive
    ResponsibleAIDashboard(rai_insights, port=PORT_RAI)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping Dashboard...")

if __name__ == "__main__":
    main()