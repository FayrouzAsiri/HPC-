# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    """Load and clean the dataset."""
    dataset = pd.read_csv(file_path)
    
    # Clean the dataset
    dataset.columns = dataset.columns.str.strip()
    columns_to_clean = ['p_min', 'p_max', 'n_min', 'n_max', 'k_min', 'k_max']
    for col in columns_to_clean:
        dataset[col] = dataset[col].astype(str).str.replace(' ppm', '').astype(float)
    
    return dataset

def encode_categorical_columns(dataset):
    """Encode the categorical columns."""
    label_encoder_tree = LabelEncoder()
    label_encoder_soil = LabelEncoder()
    
    # Encoding tree_name and soil_name into numbers for the model
    dataset['tree_name'] = label_encoder_tree.fit_transform(dataset['tree_name'])
    dataset['soil_name'] = label_encoder_soil.fit_transform(dataset['soil_name'])
    
    return dataset, label_encoder_tree, label_encoder_soil

def augment_data(row, num_variations=5):
    """Augment the dataset with variations."""
    augmented_rows = []
    for _ in range(num_variations):
        new_row = row.copy()
        for col in ['ph_min', 'ph_max', 'temperature celseus_min', 'temperature celseus_max', 
                    'p_min', 'p_max', 'n_min', 'n_max', 'k_min', 'k_max', 'Humidity%_min', 'Humidity%_max']:
            new_row[col] = row[col] * np.random.uniform(0.95, 1.05)  # Slight variation of 5%
        augmented_rows.append(new_row)
    return augmented_rows

def augment_dataset(dataset):
    """Apply augmentation to the entire dataset."""
    augmented_data = []
    for _, row in dataset.iterrows():
        augmented_data.append(row)
        augmented_data.extend(augment_data(row))
    augmented_dataset = pd.DataFrame(augmented_data)
    
    return augmented_dataset

# Main data processing function
def process_data(file_path):
    dataset = load_and_clean_data(file_path)
    
    # Drop 'tree_index' and 'soil_index' as they are not required for the prediction
    dataset = dataset.drop(columns=['tree_index', 'soil_index'])

    dataset, label_encoder_tree, label_encoder_soil = encode_categorical_columns(dataset)
    augmented_dataset = augment_dataset(dataset)
    
    return augmented_dataset, label_encoder_tree, label_encoder_soil