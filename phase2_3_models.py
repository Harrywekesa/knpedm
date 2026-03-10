import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

def load_and_preprocess_data(filepath='simulated_student_data.csv'):
    """Loads data, scales features, and splits into train/test sets."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Feature Engineering (Selecting X and y)
    X = df[['Department', 'Attendance_Pct', 'Assignment_Score', 'CAT_1_Score', 'CAT_2_Score']]
    # Convert Target 'Status' to binary (1 for Pass, 0 for Fail)
    y = df['Status'].apply(lambda x: 1 if x == 'Pass' else 0)
    
    # One-Hot Encode the 'Department' column
    X = pd.get_dummies(X, columns=['Department'], drop_first=True)
    
    # Check if we have both classes
    if len(y.unique()) < 2:
        print("WARNING: Only one class present in target variable! We need to forcefully inject some failures for realistic training.")
        # Forcefully inject some 'Fail' values for simulation robustness
        y.iloc[:30] = 0
        df.loc[:29, 'Status'] = 'Fail'
        df.loc[:29, 'Final_Exam_Score'] = np.random.uniform(20, 49, 30)
        X_cols = X.columns # Save columns before recreating X
        X = df[['Department', 'Attendance_Pct', 'Assignment_Score', 'CAT_1_Score', 'CAT_2_Score']]
        X = pd.get_dummies(X, columns=['Department'], drop_first=True)
        # Ensure injected data has the same dummy columns
        missing_cols = set(X_cols) - set(X.columns)
        for c in missing_cols:
            X[c] = False
        X = X[X_cols] # reorder
        
        print("Injected simulated failures to balance the dataset temporarily.")
    
    # 2. Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scaling (Standardization is good for Logistic Regression and general distance metrics)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use in predictions
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Trains the 3 models from the project methodology and evaluates them."""
    print("\n--- Training Models ---")
    os.makedirs('evaluation_plots', exist_ok=True)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'Accuracy': acc, 'Model': model}
        print(f"Accuracy: {acc:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save Model
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'evaluation_plots/confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close()
        
    print("\nModel Training and Evaluation Complete.")
    
    # Plot Accuracy Comparison Chart
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    accuracies = [results[n]['Accuracy'] for n in names]
    sns.barplot(x=names, y=accuracies, palette='viridis')
    plt.ylim(0, 1.1)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy Score')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('evaluation_plots/accuracy_comparison.png')
    plt.close()
    
    # Display the best model
    best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
    print(f"\n*** Best Model: {best_model_name} with Accuracy {results[best_model_name]['Accuracy']:.2f} ***")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names)
