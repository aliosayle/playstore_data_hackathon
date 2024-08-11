def find_similar_apps(app_name, n_similar=5):
    # Import necessary libraries
    import joblib
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from pathlib import Path
    
    pd.set_option('display.float_format', '{:.2f}'.format)

    # Define relative paths
    model_path = Path('models/kmeans_with_preprocessor.joblib')
    csv_path = Path('csv/df_merged_cleaned.csv')
    
    # Load the pre-trained model and preprocessor
    loaded_model = joblib.load(model_path)
    preprocessor = loaded_model['preprocessor']

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Define features to include in clustering
    features = ['rating', 'reviews', 'size', 'installs', 'price', 'average_sentiment_analysis', 'average_sentiment_subjectivity']
    
    # Create a feature matrix and apply preprocessing
    X = df[features].copy()
    X_processed = preprocessor.transform(X)

    # Check if the app_name exists
    if app_name not in df['app'].values:
        return f"App '{app_name}' not found in the dataset."
    
    # Find the app's index and feature vector
    app_index = df[df['app'] == app_name].index[0]
    app_vector = X_processed[app_index].reshape(1, -1)
    
    # Calculate cosine similarity and get indices of the most similar apps
    similarities = cosine_similarity(app_vector, X_processed).flatten()
    similar_indices = similarities.argsort()[-(n_similar + 1):-1]
    
    # Exclude the app itself and aggregate statistics
    similar_apps = df.iloc[similar_indices]
    aggregation = similar_apps[['rating', 'reviews', 'size', 'installs', 'price']].agg(['mean', 'std'])
    
    # Return similar apps and aggregated statistics
    return similar_apps[['app', 'rating', 'reviews', 'size', 'installs', 'price']], aggregation

print(find_similar_apps('instagram'))