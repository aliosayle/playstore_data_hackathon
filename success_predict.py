def predict_success(input_series):
    """
    Predicts the success of an app based on the provided input features.

    This function loads a pre-trained model from a file, converts the input features
    into a DataFrame, and uses the model to make a prediction. The prediction is
    returned as a percentage, representing the probability of success, increased by 10%.

    Args:
        input_series (dict): A dictionary containing the features for prediction. The dictionary should include:
            - 'category' (str): The category of the item.
            - 'size' (int): The size of the item.
            - 'type' (str): The type of the item (e.g., 'Free').
            - 'price' (float): The price of the item.
            - 'content rating' (str): The content rating of the item.
            - 'genres' (str): The genres associated with the item.
            - 'current ver' (str): The current version of the item.
            - 'android ver' (str): The minimum Android version required.
            - 'sentiment' (int): The sentiment score related to the item.

    Returns:
        float: The predicted probability of success of the item, as a percentage, increased by 10%.
    """
    import joblib
    import pandas as pd
    import os
    
    # Load the saved model
    model_path = os.path.join('models', 'success_prediction_model.joblib')
    loaded_model = joblib.load(model_path)

    # Create a DataFrame from the input Series
    test_data = pd.DataFrame([input_series])

    # Pass the test data through the pipeline and predict probabilities
    predictions = loaded_model.predict_proba(test_data)

    # Get the probability of the positive class (success) as a percentage
    success_probability = predictions[0][1] * 100

    # Increase the percentage by 10%
    adjusted_probability = success_probability + 10

    # Ensure the percentage does not exceed 100%
    return min(adjusted_probability, 100)

# Example usage
input_series = {
    'category': 'GAME',
    'size': 50000,
    'type': 'Free',
    'price': 0,
    'content rating': 'Everyone',
    'genres': 'Action',
    'current ver': '2.3.0',
    'android ver': '4.0 and up',
    'sentiment': 50
}

print(predict_success(input_series))
