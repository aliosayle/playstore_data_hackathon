import pandas as pd


def transform(df):

    
    def clean_text(df):
        df.columns = df.columns.str.lower()
        str_columns = df.select_dtypes(include='object').columns
        
        for col in str_columns:
            df[col] = df[col].str.lower().str.replace(' ', '_')
        
        return df
    
    def drop_na_reviews(df):
        df.dropna(subset=['translated_review'], inplace=True)
        return df


    def create_app_id_column(df, app_name_col, new_id_col):
        unique_app_names = df[app_name_col].unique()
        app_id_mapping = {app_name: idx + 1 for idx, app_name in enumerate(unique_app_names)}
        df[new_id_col] = df[app_name_col].map(app_id_mapping)
        cols = [new_id_col] + [col for col in df.columns if col not in [new_id_col]]
        df = df[cols]
        return df


    def create_reviews_summary(df, app_name_col, app_id_col, sentiment_col, subjectivity_col):
        
        summary = df.groupby([app_id_col, app_name_col]).agg(
            number_of_reviews=(app_id_col, 'count'),
            average_sentiment_analysis=(sentiment_col, 'mean'),
            average_sentiment_subjectivity=(subjectivity_col, 'mean')
        ).reset_index()
        
    
        df = pd.merge(df, summary, on=[app_id_col, app_name_col], how='left')
        
        def map_sentiment(polarity):
            if polarity > 0:
                return 'positive'
            elif polarity < 0:
                return 'negative'
            else:
                return 'neutral'
            
        def map_subjectivity(subjectivity):
            if subjectivity > 0.5:
                return 'fact'
            elif subjectivity < 0.5:
                return 'opinion'
            elif subjectivity == 0.5:
                return 'mixed'
            
        df['sentiment_category'] = df['average_sentiment_analysis'].apply(map_sentiment)
        df['subjectivity_category'] = df['average_sentiment_subjectivity'].apply(map_subjectivity)
        
        return df
    clean_text(df)
    drop_na_reviews(df)
    create_app_id_column(df,'app', 'app_id')
    create_reviews_summary(df, 'app', 'app_id', 'sentiment_polarity', 'sentiment_subjectivity')
