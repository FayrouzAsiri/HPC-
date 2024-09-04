
# Soil-Tree Recommendation System

## Project Overview

The **Soil-Tree Recommendation System** is a machine learning-based project aimed at recommending the best tree to plant based on a variety of soil and environmental parameters. The system uses the **XGBoost** algorithm, a powerful and efficient classifier, trained on augmented soil data. 

The project takes user inputs for soil characteristics (pH, temperature, humidity, nutrients, etc.), and based on these inputs, it suggests the most suitable tree species for planting. It also provides user-friendly output by displaying the names of both the tree and the soil type, rather than numeric indices.

### Motivation

Agricultural and forestry professionals, as well as hobbyists, often face challenges in selecting the right trees for specific soil types and climates. Misaligned choices can lead to poor growth or complete failure of trees. This system is designed to mitigate such risks by using historical data and machine learning to make well-informed, data-driven recommendations.

## Project Structure

```plaintext
.
├── assets/
│   └── Augmented_Soil_Dataset.csv  # The dataset used for training and recommendation
├── src/
│   ├── data_preprocessing.py       # Script to clean and augment the dataset
│   ├── model_training.py           # Script to train the model and evaluate performance
│   ├── recommendation_system.py    # Script to run the recommendation system based on user input
├── README.md                       # Project documentation
├── requirements.txt                # List of required dependencies
└── venv/                           # Virtual environment for isolating dependencies
```

## Dataset Structure

The dataset (`Augmented_Soil_Dataset.csv`) contains both soil and tree information. It is used to train the machine learning model. The structure of the dataset is as follows:

| tree_index | tree_name | ph_min | ph_max | temperature celseus_min | temperature celseus_max | p_min | p_max | n_min | n_max | k_min | k_max | Humidity%_min | Humidity%_max | soil_index | soil_name  |
|------------|-----------|--------|--------|-------------------------|-------------------------|-------|-------|-------|-------|-------|-------|----------------|---------------|------------|------------|
| 0          | Grapes    | 6.5    | 7.5    | 10                      | 20                      | 200   | 500   | 500   | 1000  | 500   | 1000  | 60             | 80            | 0          | Loam soil  |
| 1          | Apple     | 6.0    | 7.0    | 8                       | 18                      | 150   | 450   | 400   | 900   | 450   | 900   | 50             | 75            | 1          | Sandy soil |

### Dataset Columns:
- **tree_index**: Index of the tree (for internal use).
- **tree_name**: Name of the tree (used for output).
- **ph_min / ph_max**: The pH range suitable for the tree.
- **temperature celseus_min / temperature celseus_max**: The minimum and maximum temperature (in Celsius) that the tree can tolerate.
- **p_min / p_max**: Phosphorus range (in ppm) for the tree.
- **n_min / n_max**: Nitrogen range (in ppm) for the tree.
- **k_min / k_max**: Potassium range (in ppm) for the tree.
- **Humidity%_min / Humidity%_max**: The minimum and maximum humidity (%) suitable for the tree.
- **soil_index**: Index of the soil type (for internal use).
- **soil_name**: Name of the soil type (used for output).

### Dataset Augmentation:
The system uses **data augmentation** to create variations of each data point by slightly varying the numeric values for pH, temperature, and nutrient content. This augmentation helps improve the generalizability of the model and provides more robust predictions.

## How It Works

### 1. **Data Preprocessing**

The `data_preprocessing.py` script:
- Cleans the dataset by removing unnecessary characters and converting columns to the correct data types.
- Encodes categorical columns (`tree_name` and `soil_name`) into numeric values using `LabelEncoder`.
- Augments the dataset to generate new data points by applying small variations to the numeric features (e.g., pH, temperature, and nutrients).

### 2. **Model Training**

The `model_training.py` script:
- Uses the preprocessed and augmented dataset to train an XGBoost classifier.
- Splits the dataset into training and testing sets (80-20 split) to evaluate the model's accuracy.
- Saves the trained model and the label encoders (for tree and soil) in a `.pkl` file for later use in the recommendation system.

### 3. **Tree Recommendation**

The `recommendation_system.py` script:
- Takes user input for soil parameters (pH, temperature, humidity, and nutrient levels).
- Predicts the best tree to plant based on the user input using the trained XGBoost model.
- Displays the recommendation as a human-readable tree name and soil type, rather than indices.

### 4. **XGBoost Model**

**XGBoost** (Extreme Gradient Boosting) is used as the classifier in this system due to its efficiency and performance in handling tabular data. XGBoost is particularly effective for classification tasks like this, where a wide range of input features needs to be considered in making the final recommendation.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/soil-tree-recommendation-system.git
cd soil-tree-recommendation-system
```

### 2. Set Up a Virtual Environment

We recommend using a virtual environment to keep dependencies isolated.

```bash
python3 -m venv venv
source venv/bin/activate  # MacOS/Linux
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Train the Model

Before using the recommendation system, you must train the model. Ensure the dataset (`Augmented_Soil_Dataset.csv`) is located in the `assets/` directory.

To train the model, run:

```bash
python src/model_training.py
```

This will process the dataset, train the XGBoost model, and save the trained model and encoders in a file (`xgb_model.pkl`).

### 2. Run the Recommendation System

After training, you can use the recommendation system by running:

```bash
python src/recommendation_system.py
```

You will be prompted to enter the soil parameters such as pH, temperature, humidity, and nutrient levels. The system will then recommend the best tree to plant based on your input.

Example interaction:

```plaintext
Enter the soil parameters for recommendation:
pH Min: 6.5
pH Max: 7.5
Temperature Min (Celsius): 10
Temperature Max (Celsius): 20
Phosphorus Min: 200
Phosphorus Max: 500
Nitrogen Min: 500
Nitrogen Max: 1000
Potassium Min: 500
Potassium Max: 1000
Humidity Min (%): 60
Humidity Max (%): 80

Soil types available: Loam soil, Sandy soil, Clay soil
Enter the soil type: Loam soil

Recommended tree to plant: Grapes
```

## Customization

You can extend the functionality by:
1. **Adding more trees and soil types** to the dataset to improve the system's applicability.
2. **Fine-tuning the XGBoost model** with different hyperparameters to optimize accuracy.
3. **Implementing additional features**, such as supporting multiple trees for a given soil type, improving the user interface, or integrating with APIs.

## Dependencies

This project requires the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `pickle`

You can install all dependencies using the `requirements.txt` file by running:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

Contributions are welcome! If you have any improvements or bug fixes, feel free to submit a pull request. For major changes, please open an issue to discuss what you would like to change.

---

