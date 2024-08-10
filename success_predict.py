import joblib
import pandas as pd
import os

def predict_success(input_series):
    """
    Predicts the success of an app based on the provided input features.

    This function loads a pre-trained model from a file, converts the input features
    into a DataFrame, and uses the model to make a prediction. The prediction is
    returned as an integer (0 or 1), representing the success outcome.

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
        int: The predicted success of the item. The value is 0 or 1, where 1 indicates success and 0 indicates failure.
    """
    # Load the saved model
    model_path = os.path.join('models', 'success_prediction_model.joblib')
    loaded_model = joblib.load(model_path)

    # Create a DataFrame from the input Series
    test_data = pd.DataFrame([input_series])

    # Pass the test data through the pipeline and predict
    predictions = loaded_model.predict(test_data)

    # Return the first prediction as 0 or 1
    return int(predictions[0])

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
