# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

# Import the data processing function from data_preprocessing.py
from data_preprocessing import process_data

def train_model(dataset_path, model_output_path='xgb_model.pkl'):
    """Train an XGBoost model."""
    
    print("Step 1: Starting the training process...")

    # Process the dataset
    print("Step 2: Processing the dataset...")
    dataset, label_encoder_tree, label_encoder_soil = process_data(dataset_path)
    print("Step 2 Complete: Dataset has been processed.")

    # Features and target
    print("Step 3: Splitting the dataset into features (X) and target (y)...")
    X = dataset.drop(columns=['tree_name'])  # 'tree_name' is the target
    y = dataset['tree_name']  # Target variable (encoded)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Train-test split
    print("Step 4: Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Train the model using XGBoost
    print("Step 5: Training the XGBoost model...")
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Step 5 Complete: Model has been trained.")

    # Test the accuracy
    print("Step 6: Evaluating the model...")
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the model and encoders
    print(f"Step 7: Saving the trained model to {model_output_path}...")
    with open(model_output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder_tree': label_encoder_tree,
            'label_encoder_soil': label_encoder_soil
        }, f)
    print("Step 7 Complete: Model and encoders have been saved.")

# Call the train_model function to train and save the model
if __name__ == "__main__":
    print("Initializing the model training script...")
    train_model('assets/Augmented_Soil_Dataset.csv')