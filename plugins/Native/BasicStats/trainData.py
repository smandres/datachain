import math
from semantic_kernel.skill_definition import (
    sk_function,
    sk_function_context_parameter,
)
from semantic_kernel.orchestration.sk_context import SKContext
import pandas as pd
from sklearn.model_selection import train_test_split


class trainData:
    def __init__(self):
        pass
    
    @sk_function(
        description="Splits a dataset into training and testing sets",
        name="pca",
        input_description="The dataframe (mandatory) and training size (mandatory)",
    )

    def split_data(self, dataframe, train_size=0.8, random_state=None):
        """
        Split a DataFrame into training and testing datasets.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame to be split.
        - train_size (float): The proportion of the data to include in the training set (default is 0.8).
        - random_state (int or None): Seed for random number generation (optional).

        Returns:
        - tuple: A tuple containing the training DataFrame and testing DataFrame.
        """
        train_df, test_df = train_test_split(dataframe, train_size=train_size, random_state=random_state)
        return train_df, test_df