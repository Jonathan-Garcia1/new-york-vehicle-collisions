import pandas as pd
import numpy as np
import os
import requests
import math
from sklearn.model_selection import train_test_split

from sklearn.cluster import DBSCAN


from sklearn.feature_selection import SelectKBest, f_regression, RFE, chi2
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

# Importing necessary libraries for Logistic Regression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

def train_val_test_split(df):
    # Splitting the data into train, validation, and test sets with a 70-15-15 ratio
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['injuries'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['injuries'])
    return train, val, test

def create_cluster(train, val, test):

# Initialize DBSCAN
    dbscan = DBSCAN(eps=0.002, min_samples=15, metric='manhattan')

    # Prepare the data from the train set and fit the model
    injury_data_train = train[train['injuries']][['latitude', 'longitude']].values
    dbscan_labels_train = dbscan.fit_predict(injury_data_train)

    # Prepare the data from the validation and test sets
    injury_data_val = val[val['injuries']][['latitude', 'longitude']].values
    dbscan_labels_val = dbscan.fit_predict(injury_data_val)
    injury_data_test = test[test['injuries']][['latitude', 'longitude']].values
    dbscan_labels_test = dbscan.fit_predict(injury_data_test)

    # Create new DataFrames to hold the cluster labels and unique identifiers
    train_clusters = train[train['injuries']][['collision_id']].copy()
    train_clusters['cluster'] = dbscan_labels_train
    val_clusters = val[val['injuries']][['collision_id']].copy()
    val_clusters['cluster'] = dbscan_labels_val
    test_clusters = test[test['injuries']][['collision_id']].copy()
    test_clusters['cluster'] = dbscan_labels_test

    # Merge these new DataFrames back into the original train, validation, and test sets
    train = train.merge(train_clusters, on='collision_id', how='left')
    val = val.merge(val_clusters, on='collision_id', how='left')
    test = test.merge(test_clusters, on='collision_id', how='left')

    # Fill NaN cluster labels with a new cluster label that indicates 'noise' or 'unclassified'
    noise_label = -1
    train['cluster'].fillna(noise_label, inplace=True)
    val['cluster'].fillna(noise_label, inplace=True)
    test['cluster'].fillna(noise_label, inplace=True)

    # Convert cluster labels to integers
    train['cluster'] = train['cluster'].astype(int)
    val['cluster'] = val['cluster'].astype(int)
    test['cluster'] = test['cluster'].astype(int)


    # Group by cluster label and count the number of injuries in each cluster
    train_clusters['cluster_injury_count'] = train_clusters.groupby('cluster')['collision_id'].transform('count')

    # Remove duplicate cluster labels to have a unique set
    unique_train_clusters = train_clusters.drop_duplicates(subset=['cluster']).copy()

    # Sort by cluster_injury_count
    unique_train_clusters = unique_train_clusters.sort_values('cluster_injury_count')

    # Create a new column cluster_new_name based on the order of the sort
    unique_train_clusters['cluster_new_name'] = range(len(unique_train_clusters))
    # Ensure that cluster -1 should remain -1
    unique_train_clusters.loc[unique_train_clusters['cluster'] == -1, 'cluster_new_name'] = -1

    # Map the 'cluster_new_name' and 'cluster_injury_count' from 'unique_train_clusters' to the original train, validation, and test sets
    mapping_new_name = dict(unique_train_clusters[['cluster', 'cluster_new_name']].values)
    mapping_injury_count = dict(unique_train_clusters[['cluster', 'cluster_injury_count']].values)

    train['cluster_new_name'] = train['cluster'].map(mapping_new_name)
    train['cluster_injury_count'] = train['cluster'].map(mapping_injury_count)

    val['cluster_new_name'] = val['cluster'].map(mapping_new_name)
    val['cluster_injury_count'] = val['cluster'].map(mapping_injury_count)

    test['cluster_new_name'] = test['cluster'].map(mapping_new_name)
    test['cluster_injury_count'] = test['cluster'].map(mapping_injury_count)

    # Fill NaN values for 'cluster_new_name' and 'cluster_injury_count' with -1 and 0 respectively
    train['cluster_new_name'].fillna(-1, inplace=True)
    train['cluster_injury_count'].fillna(0, inplace=True)

    val['cluster_new_name'].fillna(-1, inplace=True)
    val['cluster_injury_count'].fillna(0, inplace=True)

    test['cluster_new_name'].fillna(-1, inplace=True)
    test['cluster_injury_count'].fillna(0, inplace=True)

    # Convert the new columns to integer type
    train['cluster_new_name'] = train['cluster_new_name'].astype(int)
    train['cluster_injury_count'] = train['cluster_injury_count'].astype(int)

    val['cluster_new_name'] = val['cluster_new_name'].astype(int)
    val['cluster_injury_count'] = val['cluster_injury_count'].astype(int)

    test['cluster_new_name'] = test['cluster_new_name'].astype(int)
    test['cluster_injury_count'] = test['cluster_injury_count'].astype(int)

    # Drop the original 'cluster' column and rename 'cluster_new_name' to 'cluster'
    train.drop(columns=['cluster'], inplace=True)
    val.drop(columns=['cluster'], inplace=True)
    test.drop(columns=['cluster'], inplace=True)

    train.rename(columns={'cluster_new_name': 'cluster'}, inplace=True)
    val.rename(columns={'cluster_new_name': 'cluster'}, inplace=True)
    test.rename(columns={'cluster_new_name': 'cluster'}, inplace=True)

    return train, val, test



def create_ref_date(train, val, test):
    # Create a reference date (New Year 2022)
    ref_date = pd.Timestamp('2022-01-01 00:00:00')

    # Calculate the time since the reference date in days
    train['time_since_ref_date'] = (train['crash_datetime'] - ref_date).dt.days
    val['time_since_ref_date'] = (val['crash_datetime'] - ref_date).dt.days
    test['time_since_ref_date'] = (test['crash_datetime'] - ref_date).dt.days

    # Drop 'crash_datetime', 'crash_date', 'crash_time'
    train.drop(['crash_datetime', 'crash_date', 'crash_time'], axis=1, inplace=True)
    val.drop(['crash_datetime', 'crash_date', 'crash_time'], axis=1, inplace=True)
    test.drop(['crash_datetime', 'crash_date', 'crash_time'], axis=1, inplace=True)

    # Drop 'injuries_count', 'deaths_count', 'deaths', 'collision_id
    train.drop(['injuries_count', 'deaths_count', 'deaths', 'collision_id'], axis=1, inplace=True)
    val.drop(['injuries_count', 'deaths_count', 'deaths', 'collision_id'], axis=1, inplace=True)
    test.drop(['injuries_count', 'deaths_count', 'deaths', 'collision_id'], axis=1, inplace=True)
    
    return train, val, test

def min_max_scaler (train, val, test):

    # Select only the numerical columns
    numerical_columns = train.select_dtypes(include=['int64', 'float64', 'int32'])
    # numerical_columns = train.drop(columns=['collision_id'], axis=1)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the scaler on the training data
    train[numerical_columns.columns] = scaler.fit_transform(train[numerical_columns.columns])

    # Transform the validation and test data using the same scaler
    val[numerical_columns.columns] = scaler.transform(val[numerical_columns.columns])
    test[numerical_columns.columns] = scaler.transform(test[numerical_columns.columns])
    
    return train, val, test

def hot_encode_fs(train, val, test):

    encode_cols = train.drop(columns=['on_street_name']).select_dtypes(include=['object']).columns
    # train =  train.drop(columns=['on_street_name'], axis=1)
    # val = val.drop(columns=['on_street_name'], axis=1)
    
    train_encoded = pd.get_dummies(train, columns=encode_cols)
    val_encoded = pd.get_dummies(val, columns=encode_cols)
    test_encoded = pd.get_dummies(test, columns=encode_cols)
    return train_encoded, val_encoded, test_encoded

def select_kbest(train, val):
    # Extract features and target variable from the training set
    # Here, we include both numerical and properly encoded categorical variables
    X_train = train.drop(['injuries', 'on_street_name'], axis=1)
    y_train = val['injuries']

    # Initialize SelectKBest with chi2
    selector_kbest = SelectKBest(score_func=chi2, k=5)

    # Fit SelectKBest
    selector_kbest = selector_kbest.fit(X_train, y_train)

    # Get the scores
    kbest_scores = pd.Series(selector_kbest.scores_, index=X_train.columns)
    kbest_scores
    return kbest_scores


def rfe_lasso(train):

    # Extract features and target variable from the training set
    X_train = train.drop(['injuries'], axis=1).select_dtypes(include=['int64', 'float64', 'int32'])
    y_train = train['injuries']

    # Initialize a Logistic Regression model
    model = LogisticRegression()

    # Initialize RFE with Logistic Regression
    selector_rfe = RFE(estimator=model, n_features_to_select=5)

    # Fit RFE
    selector_rfe = selector_rfe.fit(X_train, y_train)

    # Get the ranking of features
    rfe_ranking = pd.Series(selector_rfe.ranking_, index=X_train.columns)

    # Initialize and fit Logistic Regression with L1 penalty
    lasso = LogisticRegression(penalty='l1', solver='liblinear').fit(X_train, y_train)

    # Get feature importance
    lasso_importance = pd.Series(np.abs(lasso.coef_[0]), index=X_train.columns)

    # Combine all feature rankings
    feature_ranking = pd.DataFrame({
        'RFE': rfe_ranking,
        # 'SelectKBest': kbest_scores,
        'Lasso': lasso_importance
    })

    # Display the combined feature rankings
    return feature_ranking


def hot_encode_mdl(train, val, test):

    # Temporarily concatenate the train, val, and test datasets
    temp_df = pd.concat([train, val, test], keys=['train', 'val', 'test'])

    # One-hot encode the categorical columns
    encode_cols = temp_df.drop(columns=['on_street_name']).select_dtypes(include=['object']).columns
    temp_df_encoded = pd.get_dummies(temp_df, columns=encode_cols)

    # Split the data back into train, val, and test sets
    train_encoded = temp_df_encoded.loc['train']
    val_encoded = temp_df_encoded.loc['val']
    test_encoded = temp_df_encoded.loc['test']
    
    return train_encoded, val_encoded, test_encoded




def logistic_reg(train_encoded, val_encoded): 
    
        # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Model', 'Train Accuracy (%)', 'Validation Accuracy (%)'])


    # Defining the features and the target variable
    X_train = train_encoded.drop(['injuries'], axis=1)#.select_dtypes(include=['int64', 'float64', 'int32'])
    y_train = train_encoded['injuries']
    X_val = val_encoded.drop(['injuries'], axis=1)#.select_dtypes(include=['int64', 'float64', 'int32'])
    y_val = val_encoded['injuries']

    # Initializing the Logistic Regression model
    logistic_model = LogisticRegression(random_state = 42, max_iter = 500, multi_class= 'multinomial') 

    # Fitting the model
    logistic_model.fit(X_train, y_train)

    logistic_train_accuracy = round(accuracy_score(y_train, logistic_model.predict(X_train)) * 100, 1)
    logistic_val_accuracy = round(accuracy_score(y_val, logistic_model.predict(X_val)) * 100, 1)
    
    # Create a DataFrame for the results of the Logistic Regression model
    new_row = pd.DataFrame({'Model': ['Logistic Regression'], 'Train Accuracy (%)': [logistic_train_accuracy], 'Validation Accuracy (%)': [logistic_val_accuracy]})
    
    # Concatenate the new_row DataFrame to results_df
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Displaying the accuracy
    return results_df



def rand_forest(train_encoded, val_encoded):
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Model', 'Train Accuracy (%)', 'Validation Accuracy (%)'])

    # Defining the features and the target variable
    X_train = train_encoded.drop(['injuries', 'on_street_name'], axis=1)
    y_train = train_encoded['injuries']
    X_val = val_encoded.drop(['injuries', 'on_street_name'], axis=1)
    y_val = val_encoded['injuries']

    # Initializing the Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

    # Fitting the model to the training data
    rf_model.fit(X_train, y_train)

    rf_train_accuracy = round(accuracy_score(y_train, rf_model.predict(X_train)) * 100, 1)
    
    rf_val_accuracy = round(accuracy_score(y_val, rf_model.predict(X_val)) * 100, 1)
    
    # Concatenate the results as a new DataFrame row
    new_row = pd.DataFrame({'Model': ['Random Forest'], 'Train Accuracy (%)': [rf_train_accuracy], 'Validation Accuracy (%)': [rf_val_accuracy]})
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df


def gradient_boost(train_encoded, val_encoded):
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Model', 'Train Accuracy (%)', 'Validation Accuracy (%)'])

    # Defining the features and the target variable
    X_train = train_encoded.drop(['injuries', 'on_street_name'], axis=1)
    y_train = train_encoded['injuries']
    X_val = val_encoded.drop(['injuries', 'on_street_name'], axis=1)
    y_val = val_encoded['injuries']

    # Initializing the Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)

    # Fitting the model to the training data
    gb_model.fit(X_train, y_train)

    gb_train_accuracy = round(accuracy_score(y_train, gb_model.predict(X_train)) * 100, 1)
    gb_val_accuracy = round(accuracy_score(y_val, gb_model.predict(X_val)) * 100, 1)

    # Create a DataFrame for the results of the Gradient Boosting model
    new_row = pd.DataFrame({'Model': ['Gradient Boosting'], 'Train Accuracy (%)': [gb_train_accuracy], 'Validation Accuracy (%)': [gb_val_accuracy]})

    # Concatenate the new_row DataFrame to results_df
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df

# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def test_accuracy(train_encoded, classifier_type):
    # Defining the features and the target variable
    X_train = train_encoded.drop(['injuries'], axis=1).select_dtypes(include=['int64', 'float64', 'int32'])
    y_train = train_encoded['injuries']

    # Initialize the specified classifier
    if classifier_type == 'LogisticRegression':
        classifier = LogisticRegression(random_state=42, max_iter=500, multi_class='multinomial')
    elif classifier_type == 'RandomForestClassifier':
        classifier = RandomForestClassifier(random_state=42, n_estimators=100)
    elif classifier_type == 'GradientBoostingClassifier':
        classifier = GradientBoostingClassifier(random_state=42, n_estimators=100)
    else:
        raise ValueError("Invalid classifier_type. Please choose one of: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier.")

    # Fitting the model
    classifier.fit(X_train, y_train)

    # Calculate the accuracy score on the training data
    accuracy_results = accuracy_score(y_train, classifier.predict(X_train))

    return accuracy_results