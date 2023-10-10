import pandas as pd
import numpy as np
import os
import requests
import math
import env
import wrangle as w
import explore as exp
import datetime
from scipy.stats import chi2_contingency
# Importing necessary libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2






def create_info_dataframe(df):
    # Initialize lists to store information for each column
    column_names = []
    null_counts = []
    null_percentages = []
    zero_counts = []
    blank_counts = []
    unique_value_counts = []
    data_types = []

    # Iterate through the columns of the DataFrame
    for column in df.columns:
        column_names.append(column)
        null_count = df[column].isnull().sum()
        null_counts.append(null_count)
        total_count = len(df)
        null_percentage = (null_count / total_count) * 100
        null_percentages.append(np.round(null_percentage))
        zero_count = (df[column] == 0).sum()
        zero_counts.append(zero_count)
        blank_count = (df[column] == " ").sum()
        blank_counts.append(blank_count)
        unique_count = df[column].nunique()
        unique_value_counts.append(unique_count)
        data_type = df[column].dtype
        data_types.append(data_type)

    # Create the information DataFrame
    info_df = pd.DataFrame({
        "Column": column_names,
        "Null_Count": null_counts,
        "Null_Percentage": null_percentages,
        "Zero_Count": zero_counts,
        "Blank_Count": blank_counts,
        "Unique_Values": unique_value_counts,
        "Data_Type": data_types
    })

    return info_df


# Modified function to conduct Chi-Square Test and interpret results
def chi_square_test(dataframe, target_var, category_var):
    # Create a contingency table
    contingency_table = pd.crosstab(dataframe[category_var], dataframe[target_var])
    
    # Conduct the Chi-Square Test of Independence
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    
    # Calculate r value (effect size measure for chi-square, known as Cramér's V)
    r_value = (chi2_stat / (len(dataframe) * (min(contingency_table.shape) - 1))) ** 0.5
    
    # Print p-value and r-value
    print(f"p-value: {p_value}")
    print(f"r-value: {r_value:.2f}")
    
    # Interpretation of p-value and r-value
    print("\nInterpretation:")
    if p_value < 0.05:
        print("Reject the null hypothesis.")
    else:
        print("Fail to reject the null hypothesis.")
    
    if r_value < 0.1:
        print("The r-value suggests a small effect size.")
    elif 0.1 <= r_value < 0.3:
        print("The r-value suggests a medium effect size.")
    else:
        print("The r-value suggests a large effect size.")
