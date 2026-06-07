import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_and_clean_data(file_path):
    print("--- 1. Loading and Cleaning Data ---")
    df = pd.read_csv(file_path)
    # Binary encoding for Target
    df['Is Acquired'] = df['Exit Status'].apply(lambda x: 1 if x == 'Acquired' else 0)
    print(f"Dataset Loaded Successfully. Shape: {df.shape}")
    return df

def run_statistical_test(df):
    print("\n--- 2. Running T-Test ---")
    acquired_val = df[df['Exit Status'] == 'Acquired']['Valuation (M USD)']
    private_val = df[df['Exit Status'] == 'Private']['Valuation (M USD)']
    t_stat, p_val = ttest_ind(acquired_val, private_val, equal_var=False)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    return p_val

def train_ml_model(df):
    print("\n--- 3. Training Random Forest Classifier ---")
    # One-hot encoding categorical variables
    df_encoded = pd.get_dummies(df, columns=['Industry', 'Region'], drop_first=True)
    
    # Drop irrelevant columns for features
    X = df_encoded.select_dtypes(include='number').drop(columns=['Is Acquired', 'Year Founded', 'Profitable'])
    y = df_encoded['Is Acquired']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Balanced Random Forest to prevent majority class bias
    clf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
    print(classification_report(y_test, y_pred))

if _name_ == "_main_":
    # Adjusted path assuming script runs from the repository root
    csv_path = "data/startup_data (1).csv"
    
    try:
        startup_df = load_and_clean_data(csv_path)
        run_statistical_test(startup_df)
        train_ml_model(startup_df)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Please check file locations.")
