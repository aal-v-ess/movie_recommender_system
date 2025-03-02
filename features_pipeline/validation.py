import pandas as pd
import great_expectations as ge


def validate_features(df: pd.DataFrame):
    """Validate features with Great Expectations"""

    # Data quality checks with great expectations
    ge_df = ge.from_pandas(df)
    ge_df["user_item_rating"] = ge_df["userId"].astype(str) + "_" + ge_df["movieId"].astype(str) + "_" + ge_df["rating"].astype(str)

    # Define expectations
    expectations = [
        ge_df.expect_column_values_to_not_be_null("userId"),
        ge_df.expect_column_values_to_not_be_null("movieId"),
        ge_df.expect_column_values_to_not_be_null("rating"),
        ge_df.expect_column_values_to_not_be_null("year"),
        ge_df.expect_column_values_to_not_be_null("title"),
        ge_df.expect_column_values_to_not_be_null("genres"),
        ge_df.expect_column_values_to_be_between("rating", min_value=0, max_value=5),
        ge_df.expect_column_values_to_be_between("year", min_value=1900, max_value=2100),
        ge_df.expect_column_values_to_not_be_null("title"),
        ge_df.expect_column_values_to_match_regex("genres", r"^[\w, ]*$", mostly=0.8),
        ge_df.expect_table_row_count_to_be_between(1, len(df)),
        ge_df.expect_column_values_to_be_unique("user_item_rating")
    ]
    
    # Check if all validations passed
    passed = all(expectation.success for expectation in expectations)
    
    if not passed:
        failed = [exp for exp in expectations if not exp.success]
        print(f"Validation failed: {failed}")
    
    return passed