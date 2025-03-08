import sys
import os
import fire

from read import ReadData
from transform import EmbeddingGenerator
from training import EnhancedRecommendationPreComputer

sys.path.append(os.path.abspath("..\\"))
print(os.path.abspath("..\\"))
from utils import get_logger, load_config

logger = get_logger(__name__)


def run_training_pipeline():
    """
    """

    # Load the config
    config = load_config("..\\config.yaml")

    # Read
    df = ReadData(
        feature_store_name=config["feature_store"]["name"], 
        feature_store_key=config["HOPSWORKS_API_KEY"], 
        feature_store_version=config["feature_store"]["version"]
    ).read_and_update_data()
    df.columns = ['user_id', 'item_id', 'rating', 'user_movie_tags', 'title', 'genres', 'year']
    items_df = ReadData.transform_items_df(df)
    user_interactions_df = ReadData.transform_user_df(df)

    # Transform
    embedding_generator = EmbeddingGenerator()
    # Generate item embeddings
    item_embeddings = embedding_generator.create_item_embeddings(items_df)
    # Generate user embeddings
    user_embeddings = embedding_generator.create_user_embeddings(user_interactions_df)

    # Train
    computer = EnhancedRecommendationPreComputer(
        items_df=items_df,
        user_profiles_df=user_interactions_df,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        content_weight = 0.3,
        collaborative_weight = 0.7,
        diversity_weight = 0.05,
        base_score = 0.05,
        n_precomputed_recs = 100,
        similar_users = 100,
    )

    # Compute all recommendations
    computer.compute_all_recommendations()

    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    fire.Fire(run_training_pipeline())