import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# --- 1. CONFIGURATION ---
# Path to the dataset files created by the previous script
REAL_CSV_PATH = 'output/dataset_subset/real.csv'
FAKE_CSV_PATH = 'output/dataset_subset/fake.csv'

# Path where the trained model will be saved
MODEL_SAVE_PATH = 'model/deepfake_detector_model.joblib'
# --- END OF CONFIGURATION ---


def load_and_prepare_data(real_path, fake_path):
    """Loads, combines, and prepares the real and fake data for training."""
    print("Loading and preparing data...")
    
    # Check if data files exist
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print("\n--- ERROR ---")
        print(f"Dataset files not found. Please ensure '{real_path}' and '{fake_path}' exist.")
        print("You need to run the `create_training_dataset.py` script first.")
        return None, None

    # Load the datasets
    df_real = pd.read_csv(real_path)
    df_fake = pd.read_csv(fake_path)

    # Add the 'label' column: 0 for real, 1 for fake
    df_real['label'] = 0
    df_fake['label'] = 1

    # Combine into a single dataframe
    df_combined = pd.concat([df_real, df_fake], ignore_index=True)

    # Handle any potential missing values (e.g., from failed HR calculations)
    # Filling with 0 is a simple and often effective strategy
    df_combined = df_combined.fillna(0)

    print(f"Data loaded successfully. Total samples: {len(df_combined)}")
    print(f"Real samples: {len(df_real)}, Fake samples: {len(df_fake)}")

    # Separate features (X) from the target label (y)
    X = df_combined.drop(columns=['path', 'label'])
    y = df_combined['label']

    return X, y


def plot_confusion_matrix(y_true, y_pred):
    """Generates and displays a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Real', 'Predicted Fake'], 
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def train_and_evaluate(X, y):
    """Splits data, trains the LightGBM model, and evaluates its performance."""
    print("\nSplitting data into training and testing sets (80/20)...")
    
    # Split data, stratifying by 'y' ensures the same proportion of labels in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
    print("\nTraining LightGBM model...")

    # Initialize the LightGBM Classifier
    # Default parameters are often very strong, but can be tuned
    model = lgb.LGBMClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    print("Model training complete. Evaluating performance on the test set...")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # --- Performance Evaluation ---
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy on Test Set: {accuracy:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    return model

def save_model(model, path):
    """Saves the trained model to a file."""
    print(f"\nSaving trained model to '{path}'...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print("Model saved successfully.")


if __name__ == '__main__':
    # 1. Load and prepare the data
    features, labels = load_and_prepare_data(REAL_CSV_PATH, FAKE_CSV_PATH)
    
    if features is not None:
        # 2. Train and evaluate the model
        trained_model = train_and_evaluate(features, labels)
        
        # 3. Save the final model for future use
        save_model(trained_model, MODEL_SAVE_PATH)
        
        print("\nProcess complete. You can now use the saved model for predictions.")