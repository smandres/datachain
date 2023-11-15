import math
from semantic_kernel.skill_definition import (
    sk_function,
    sk_function_context_parameter,
)
from semantic_kernel.orchestration.sk_context import SKContext
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self):
        pass
    
    @sk_function(
        description="Calculates the slope of a linear regression line",
        name="pca",
        input_description="The dataframe (mandatory) and the number of components to retain (optional)",
    )
    def pca(dataframe, n_components=None):
        """
        Perform PCA on a DataFrame.

        Parameters:
        - dataframe: The input DataFrame with numeric columns.
        - n_components: The number of principal components to retain (optional).

        Returns:
        - pca_result: A DataFrame containing the PCA results.
        """
        # Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(dataframe)

        # Create a PCA object
        pca = PCA(n_components=n_components)

        # Fit the PCA model to the standardized data
        principal_components = pca.fit_transform(standardized_data)

        # Create a DataFrame to store the PCA results
        pca_result = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

        return pca_result