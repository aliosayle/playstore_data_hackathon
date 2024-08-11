def find_similar_apps(user_input_vector, n_similar=5):
    import json
    import numpy as np
    import joblib
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from pathlib import Path
    # Define relative paths
    model_path = Path('models/kmeans_with_preprocessor.joblib')
    csv_path = Path('csv/df_merged_cleaned.csv')
    csv_path_reviews=Path('csv/googleplaystore_user_reviews.csv')
    
    # Load the pre-trained model and preprocessor
    loaded_model = joblib.load(model_path)
    preprocessor = loaded_model['preprocessor']

    # Load the dataset
    df = pd.read_csv(csv_path)
    df_reviews=pd.read_csv( csv_path_reviews)
    # Define features to include in clustering
    features = ['rating', 'reviews', 'size', 'installs', 'price', 'average_sentiment_analysis', 'average_sentiment_subjectivity']
    
    # Create a feature matrix and apply preprocessing
    X = df[features].copy()
    X_processed = preprocessor.transform(X)

    # Convert user input vector to DataFrame
    user_input_vector_df = pd.DataFrame(user_input_vector, columns=features)
    user_input_vector_processed = preprocessor.transform(user_input_vector_df)
    
    # Convert the list to a numpy array and reshape it to match the expected input format
    app_vector = np.array(user_input_vector_processed).reshape(1, -1)
    
    # Calculate cosine similarity and get indices of the most similar apps
    similarities = cosine_similarity(app_vector, X_processed).flatten()
    similar_indices = similarities.argsort()[-(n_similar + 1):-1]
    
    # Exclude the app itself and aggregate statistics
    similar_apps = df.iloc[similar_indices]
    aggregation = similar_apps[['rating', 'reviews', 'size', 'installs', 'price']].agg(['mean', 'std'])
    
    # Convert DataFrames to JSON format
    similar_apps_json = similar_apps[['app', 'rating', 'reviews', 'size', 'installs', 'price']].to_dict(orient='records')
    aggregation_json = aggregation.to_dict(orient='index')
    
    # Return JSON-formatted results
    return json.dumps({
        'similar_apps': similar_apps_json,
        'aggregation': aggregation_json
    })

# Define a sample user input vector
user_input_vector = [{
    'rating': 4.1,
    'reviews': 190,
    'size': 24,
    'installs': 10500,
    'price': 1.79,
    'average_sentiment_analysis': 0.25,
    'average_sentiment_subjectivity': 0.15
}]

# Test the function
result_json = find_similar_apps(user_input_vector)
print(result_json)