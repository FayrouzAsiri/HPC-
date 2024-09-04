# recommendation_system.py

import pandas as pd
import pickle

# Function to recommend tree based on user input
def recommend_tree(model_path='xgb_model.pkl'):
    """Load the model and recommend a tree based on user input."""
    
    # Load the model and encoders
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        label_encoder_tree = data['label_encoder_tree']
        label_encoder_soil = data['label_encoder_soil']

    # Get user input
    print("\nEnter the soil parameters for recommendation:")
    ph_min = float(input("pH Min: "))
    ph_max = float(input("pH Max: "))
    temperature_min = float(input("Temperature Min (Celsius): "))
    temperature_max = float(input("Temperature Max (Celsius): "))
    p_min = float(input("Phosphorus Min: "))
    p_max = float(input("Phosphorus Max: "))
    n_min = float(input("Nitrogen Min: "))
    n_max = float(input("Nitrogen Max: "))
    k_min = float(input("Potassium Min: "))
    k_max = float(input("Potassium Max: "))
    humidity_min = float(input("Humidity Min (%): "))
    humidity_max = float(input("Humidity Max (%): "))
    
    # Soil type input
    soil_types = label_encoder_soil.classes_
    
    # Convert soil types to string for display
    soil_types_str = [str(soil_type) for soil_type in soil_types]
    print("\nSoil types available:", ", ".join(soil_types_str))

    soil_type_input = input("Enter the soil type: ")
    soil_type_encoded = label_encoder_soil.transform([soil_type_input])[0]

    # Prepare input for prediction (ensure the number of columns matches the model)
    user_input = pd.DataFrame([[
        ph_min, ph_max, temperature_min, temperature_max,
        p_min, p_max, n_min, n_max, k_min, k_max,
        humidity_min, humidity_max, soil_type_encoded
    ]], columns=model.feature_names_in_)  # Using the correct number of columns

    # Print the user input in a readable format
    print("\nUser input for prediction:")
    print(f"  pH Min: {ph_min}")
    print(f"  pH Max: {ph_max}")
    print(f"  Temperature Min (Celsius): {temperature_min}")
    print(f"  Temperature Max (Celsius): {temperature_max}")
    print(f"  Phosphorus Min: {p_min}")
    print(f"  Phosphorus Max: {p_max}")
    print(f"  Nitrogen Min: {n_min}")
    print(f"  Nitrogen Max: {n_max}")
    print(f"  Potassium Min: {k_min}")
    print(f"  Potassium Max: {k_max}")
    print(f"  Humidity Min (%): {humidity_min}")
    print(f"  Humidity Max (%): {humidity_max}")
    print(f"  Soil Type: {soil_type_input}")

    # Predict and decode the result
    prediction = model.predict(user_input)[0]
    
    # Get the recommended tree name instead of index
    recommended_tree = label_encoder_tree.inverse_transform([prediction])[0]
    
    # Print the result with the name of the tree and soil
    print(f"\nRecommended tree to plant: {recommended_tree}")
    print(f"  Soil Type: {soil_type_input}")

# Call the recommendation function
if __name__ == "__main__":
    recommend_tree()