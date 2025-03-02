from read import read_data
from transform import pipeline_transform
from validation import validate_features
from feature_store import register_features
from utils import get_logger
from settings import DATA_PATH


def run_features_pipeline(data_path: str, drop_columns: list = None):
    """
    Run the features pipeline
    """
    logger = get_logger(__name__)

    # Read data
    df_ratings = read_data(DATA_PATH, "ratings.csv", drop_columns)
    df_movies = read_data(DATA_PATH, "movies.csv", drop_columns)
    df_tags = read_data(DATA_PATH, "tags.csv", drop_columns)

    # Transform data
    df_transform = pipeline_transform(df_ratings, df_movies, df_tags)

    # Validate features
    if validate_features(df_transform):
        # Register to feature store with version control
        feature_group = register_features(df_transform)
        
        print(f"Feature pipeline complete. Created feature group version {feature_group.version}")
    else:
        print("Feature validation failed. Features not registered.")

    logger.info("Features pipeline completed successfully.")
    return df_ratings
