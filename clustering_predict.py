import joblib
import numpy as np
import pandas as pd

# Load the combined file
loaded_model = joblib.load('models/kmeans_with_preprocessor.joblib')

# Access the KMeans model and preprocessor
kmeans_loaded = loaded_model['kmeans']
preprocessor = loaded_model['preprocessor']

data = {
    'rating': np.random.uniform(1.0, 5.0),  # Rating between 1.0 and 5.0
    'reviews': np.random.randint(0, 10000),  # Random number of reviews between 0 and 10000
    'size': np.random.uniform(1.0, 100.0),  # Random size between 1.0 MB and 100.0 MB
    'installs': np.random.randint(100, 1000000),  # Random installs between 100 and 1,000,000
    'price': np.random.uniform(0.0, 50.0),  # Random price between 0.0 and 50.0 USD
    'average_sentiment_analysis': np.random.uniform(-1.0, 1.0),  # Sentiment analysis between -1.0 and 1.0
    'average_sentiment_subjectivity': np.random.uniform(0.0, 1.0)  # Sentiment subjectivity between 0.0 and 1.0
}

# Create DataFrame
df = pd.DataFrame([data])

X_new_processed = preprocessor.transform(data)



