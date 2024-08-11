def predict_success(input_series):
    """
    Predicts the success of an app based on the provided input features.

    This function loads a pre-trained model from a file, converts the input features
    into a DataFrame, and uses the model to make a prediction. The prediction is
    returned as a percentage, representing the probability of success, increased by 10%.
    Additionally, it returns tips on why the prediction might be low.

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
        tuple: A tuple containing the predicted probability of success of the item, as a percentage,
               and a list of tips on why the prediction might be low.
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
    adjusted_probability = min(adjusted_probability, 100)

    # Analyze the input features and provide tips if the success probability is low
    tips = []

    if input_series['price'] > 0:
        tips.append("Consider lowering the price or offering a free version to attract more users.")
    
    if input_series['sentiment'] < 60:
        tips.append("Improve user sentiment by addressing negative reviews and enhancing the user experience.")

    if input_series['size'] > 100000:
        tips.append("The app size is large, which might discourage users from downloading it.")

    if input_series['android ver'] != 'Varies with device' and input_series['android ver'] > '4.0':
        tips.append("Consider supporting older Android versions to reach a broader audience.")

    if input_series['genres'] not in ['Action', 'Puzzle', 'Casual', 'Strategy']:
        tips.append(f"Consider focusing on popular genres. Current genre '{input_series['genres']}' might not be as appealing.")

    # If the adjusted probability is less than 50%, provide a general tip
    if adjusted_probability < 50:
        tips.append("Overall success prediction is low. Consider re-evaluating the app's features and market strategy.")

    return adjusted_probability, tips

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

probability, tips = predict_success(input_series)
print(f"Predicted Success Probability: {probability}%")
print("Tips:")
for tip in tips:
    print(f"- {tip}")
