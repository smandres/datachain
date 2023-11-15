import math
from semantic_kernel.skill_definition import (
    sk_function,
    sk_function_context_parameter,
)
from semantic_kernel.orchestration.sk_context import SKContext

class LinearReg:
    def __init__(self):
        pass
    
    @sk_function(
        description="Calculates the slope of a linear regression line",
        name="Slope",
        input_description="Two lists of numbers representing the x and y values",
    )
    def slope(self, x_values: list, y_values: list) -> float:
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        numerator = sum([(x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n)])
        denominator = sum([(x_values[i] - x_mean) ** 2 for i in range(n)])
        return numerator / denominator
    
    @sk_function(
        description="Calculates the y-intercept of a linear regression line",
        name="YIntercept",
        input_description="Two lists of numbers representing the x and y values",
    )
    def y_intercept(self, x_values: list, y_values: list) -> float:
        slope = self.slope(x_values, y_values)
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)
        return y_mean - slope * x_mean
    
    @sk_function(
        description="Calculates the correlation coefficient of a linear regression line",
        name="CorrelationCoefficient",
        input_description="Two lists of numbers representing the x and y values",
    )
    def correlation_coefficient(self, x_values: list, y_values: list) -> float:
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        numerator = sum([(x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n)])
        denominator = math.sqrt(sum([(x_values[i] - x_mean) ** 2 for i in range(n)]) * sum([(y_values[i] - y_mean) ** 2 for i in range(n)]))
        return numerator / denominator
    
    @sk_function(
        description="Calculates the predicted y value for a given x value on a linear regression line",
        name="PredictedY",
        input_description="Two lists of numbers representing the x and y values, and a number representing the x value to predict the y value for",
    )
    def predicted_y(self, x_values: list, y_values: list, x: float) -> float:
        slope = self.slope(x_values, y_values)
        y_intercept = self.y_intercept(x_values, y_values)
        return slope * x + y_intercept
