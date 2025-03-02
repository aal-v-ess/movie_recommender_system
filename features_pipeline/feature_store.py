import hopsworks
import pandas as pd
from datetime import datetime

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()


def register_features(features_df, version=None):
    """Register features to Hopsworks Feature Store with version control"""
    # Check if feature group exists
    try:
        existing_fg = fs.get_feature_group("movie_rec_system")
        # If we want a new version, increment the version number
        if version is None:
            version = existing_fg.version + 1
    except:
        # Feature group doesn't exist yet, start with version 1
        version = 1 if version is None else version
    
    # Create or get feature group
    feature_group = fs.create_feature_group(
        name="movie_rec_system",
        version=version,
        description=f"Movie recommender system features v{version} created on {datetime.now().strftime('%Y-%m-%d')}",
        primary_key=["userId"],
        online_enabled=True,
        statistics_config={"enabled": True, "histograms": True},
        event_time="last_active"  # Enable time travel with event time
    )
    
    # Insert data
    feature_group.insert(features_df)
    
    # Log metadata about this version
    feature_group.add_tag({"version_info": f"Added features: {', '.join(features_df.columns)}"})
    
    print(f"Successfully registered features as version {version}")
    return feature_group