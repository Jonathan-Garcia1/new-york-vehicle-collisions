import pandas as pd
import numpy as np
import os
import requests
import datetime

def get_data(year, app_token, max_observations=None):
    # Define the base API URL
    base_url = 'https://data.cityofnewyork.us/resource/h9gi-nx95.json'

    # Check if a CSV file with the specified year already exists
    csv_filename = f'nyc_collisions_{year}.csv'
    if os.path.isfile(csv_filename):
        print(f"CSV file for {year} already exists. Loading data from the CSV.")
        df = pd.read_csv(csv_filename)
        return df

    # Initialize an empty list to store all data
    all_data = []

    # Set the initial offset to 0 and the page size to 1000
    offset = 0
    page_size = 1000

    while max_observations is None or len(all_data) < max_observations:
        # Calculate the remaining observations to retrieve
        remaining_observations = max_observations - len(all_data) if max_observations is not None else page_size

        # Calculate the actual page size for this request
        actual_page_size = min(page_size, remaining_observations)

        # Construct the URL with the app token, date filter, offset, and page size
        url = f'{base_url}?$where=crash_date between "{year}-01-01" and "{year}-12-31"&$$app_token={app_token}&$offset={offset}&$limit={actual_page_size}'

        # Send an HTTP GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()  # Convert JSON response to Python data
            if len(data) == 0:
                break  # No more data, exit the loop
            all_data.extend(data)  # Add the data to the list
            offset += actual_page_size  # Increment the offset for the next request
        else:
            print(f"Failed to retrieve data for {year}. Status code: {response.status_code}")
            return None  # Exit the function with None if data retrieval fails

        if max_observations is not None and len(all_data) >= max_observations:
            break  # Stop if the maximum number of observations has been reached

    # Create a DataFrame using pandas
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file for easy access
    df.to_csv(csv_filename, index=False)

    print(f"Data for {year} retrieved and saved to {csv_filename}.")

    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def initial_reorder_cols(df):
    col_order = ['crash_datetime', 'crash_date', 'crash_time', 'collision_id',
     'latitude', 'longitude', 'location', 
     'on_street_name', 'cross_street_name', 'off_street_name',
     'borough', 'zip_code',
     'number_of_persons_injured', 'number_of_persons_killed', 
     'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
     'number_of_cyclist_injured', 'number_of_cyclist_killed', 
     'number_of_motorist_injured', 'number_of_motorist_killed', 
     'vehicle_type_code1', 'contributing_factor_vehicle_1', 
     'vehicle_type_code2', 'contributing_factor_vehicle_2',
     'vehicle_type_code_3', 'contributing_factor_vehicle_3', 
     'vehicle_type_code_4', 'contributing_factor_vehicle_4', 
     'vehicle_type_code_5', 'contributing_factor_vehicle_5']
     
    # Reorder the columns based on the specified order
    df = df[col_order]
    
    return df
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def date_time(df):

    # Convert crash_date and crash_time to datetime first
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df['crash_time'] = pd.to_datetime(df['crash_time'], format='%H:%M').dt.time

    # Combine crash_date and crash_time into a new column crash_datetime
    df['crash_datetime'] = pd.to_datetime(df['crash_date'].dt.strftime('%Y-%m-%d') + ' ' + df['crash_time'].astype(str))

    # # Convert 'crash_date' and 'crash_time' to datetime and combine them into a single column
    # df['crash_datetime'] = pd.to_datetime(df['crash_date'].str[:10] + ' ' + df['crash_time'], format='%Y-%m-%d %H:%M')
    # df['crash_date'] = pd.to_datetime(df['crash_date'])
    return df
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def initial_drop(df):
    df.drop(columns=['location'], inplace=True)
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def drop_3plus(df):
    # Create a mask where we filter out any observation with a value in vehicle 3 +
    # This will leave us with only the observations that have 2 vehicles involved
    condition = (df['vehicle_type_code_3'].isnull() &
                df['vehicle_type_code_4'].isnull() &
                df['vehicle_type_code_5'].isnull() &
                df['contributing_factor_vehicle_3'].isnull()
                )

    # Apply the condition to filter the DataFrame
    df = df[condition]

    # Reset the index if needed
    df.reset_index(drop=True, inplace=True)

    df = df.drop(columns=['vehicle_type_code_3', 'vehicle_type_code_4', 'vehicle_type_code_5', 'contributing_factor_vehicle_3', 'contributing_factor_vehicle_4', 'contributing_factor_vehicle_5' ])
    
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def single_coll(df):
    # filter only keep observations with a value in vehicle_type_code1 and 2
    # this will leave us with collisions of 2 vehicles only. 
    condition = (df['vehicle_type_code1'].notnull() &
                df['vehicle_type_code2'].notnull())
    df = df[condition]

    # Reset the index if needed
    df.reset_index(drop=True, inplace=True)
    
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def motorist_only(df):
    # Filter the DataFrame to exclude collisions involving pedestrians or cyclists
    df = df[(df['number_of_pedestrians_injured'] == 0) & (df['number_of_pedestrians_killed'] == 0) & (df['number_of_cyclist_injured'] == 0) & (df['number_of_cyclist_killed'] == 0)]
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def second_drop(df):
    df = df.drop(columns=['cross_street_name', 'off_street_name', 'number_of_pedestrians_injured', 'number_of_pedestrians_killed', 'number_of_cyclist_injured', 'number_of_cyclist_killed'])
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def null_zero_loc(df):
    df.loc[df['latitude'] == 0.0, ['latitude', 'longitude', 'location']] = [np.nan, np.nan, np.nan]
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def fill_geo(df):
    geo_filled = pd.read_csv('final_geo_filled.csv', index_col=0)
    
    # List of unique collision_ids in filled_df
    collision_ids_to_remove = geo_filled['collision_id'].unique()

    # Remove rows from first_run where collision_id matches any value in collision_ids_to_remove
    df = df[~df['collision_id'].isin(collision_ids_to_remove)]
    
    # Concatenate the two dataframes
    df = pd.concat([geo_filled, df], ignore_index=True)
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def fill_na(df):

    df['contributing_factor_vehicle_2'] = df['contributing_factor_vehicle_2'].fillna('Unspecified')

    df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].fillna('Unspecified')
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def drop_na(df):
    df = df.dropna(subset=['latitude', 'longitude'])
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def injuries_build(df):
    # Create 'injuries_count' column based on the higher of the two injury columns
    df['injuries_count'] = df[['number_of_persons_injured', 'number_of_motorist_injured']].max(axis=1)

    # Create 'deaths_count' column based on the higher of the two death columns
    df['deaths_count'] = df[['number_of_persons_killed', 'number_of_motorist_killed']].max(axis=1)

    # Create 'injuries' column based on 'injuries_count'
    df['injuries'] = df['injuries_count'] > 0

    # Create 'deaths' column based on 'deaths_count'
    df['deaths'] = df['deaths_count'] > 0

    df =  df.drop(['number_of_persons_injured', 'number_of_motorist_injured', 'number_of_persons_killed', 'number_of_motorist_killed'], axis=1)
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def lower_case(df):
    # Update numerical_columns to include columns of type datetime.datetime
    numerical_columns = [col for col in df.columns if isinstance(df[col][0], (np.integer, float, datetime.datetime, datetime.time, bool)) or df[col].dtype == bool]
    numerical_columns
    # Lowercase the values in non-numerical columns
    for column in df.columns:
        if column not in numerical_columns:
            df[column] = df[column].str.lower()
    return df
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
# Load the CSV file back into a DataFrame

def map_categories(df):
    loaded_type_to_cat_df = pd.read_csv('type_to_cat_df.csv')

    # Convert the DataFrame back to a dictionary
    loaded_type_to_cat = dict(zip(loaded_type_to_cat_df['vehicle_types'], loaded_type_to_cat_df['category']))


    # Map the loaded dictionary onto the DataFrame columns to create new columns
    df['vehicle_1_category'] = df['vehicle_type_code1'].map(loaded_type_to_cat)
    df['vehicle_2_category'] = df['vehicle_type_code2'].map(loaded_type_to_cat)


    # Replace NaN values with 'Other' in the columns before mapping
    df['vehicle_1_category'].fillna('other', inplace=True)
    df['vehicle_2_category'].fillna('other', inplace=True)
    
    # Load the CSV file back into a DataFrame
    factors_df_loaded = pd.read_csv('factors_df.csv')

    # Convert the DataFrame back to a dictionary
    factors_to_cat = dict(zip(factors_df_loaded['factors'], factors_df_loaded['factors_category']))

    # Convert the DataFrame back to a dictionary
    factors_to_subcat = dict(zip(factors_df_loaded['factors'], factors_df_loaded['factors_subcat']))

    # Map the dictionary onto the DataFrame columns to create new columns
    df['factors_category_vehicle_1'] = df['contributing_factor_vehicle_1'].map(factors_to_cat)
    df['factors_category_vehicle_2'] = df['contributing_factor_vehicle_2'].map(factors_to_cat)
    df['factors_subcat_vehicle_1'] = df['contributing_factor_vehicle_1'].map(factors_to_subcat)
    df['factors_subcat_vehicle_2'] = df['contributing_factor_vehicle_2'].map(factors_to_subcat)

    return df


def datetime_features(df):
    # Convert 'crash_datetime' to a datetime object
    df['crash_datetime'] = pd.to_datetime(df['crash_datetime'])

    # Extract time-based features from 'crash_datetime'
    df['hour_of_day'] = df['crash_datetime'].dt.hour
    df['day_of_week'] = df['crash_datetime'].dt.day_name()
    df['month'] = df['crash_datetime'].dt.month_name()

    # Create a 'daylight' column, set to True if the hour is between 06 and 19, otherwise False
    df['daylight'] = (df['hour_of_day'] >= 6) & (df['hour_of_day'] <= 19)

    # the 'daylight_day_of_week' column to use 'Day' and 'Night' instead of True and False
    df['daylight_day_of_week'] = df['day_of_week'] + '_' + df['daylight'].map({True: 'Day', False: 'Night'})
    
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

# Define the ranking dictionary for the strength of each factor subcategory
factor_strength_ranking = {
    'substance abuse': 1,
    'driving violations': 2,
    'distraction': 3,
    'health-related': 4,
    'mechanical': 5,
    'environmental': 6,
    'unspecified': 7  # Adding 'unspecified' with the lowest rank
}

# Create a new column to store the combined factors based on the rules
def combine_factors(row):
    factor_1 = row['factors_subcat_vehicle_1']
    factor_2 = row['factors_subcat_vehicle_2']
    
    # Rule 1: If factor_1 is not 'unspecified' and factor_2 is 'unspecified', use factor_1
    if factor_1 != 'unspecified' and factor_2 == 'unspecified':
        return factor_1
    
    # Rule 2: If both are 'unspecified', then the new column should also show 'unspecified'
    if factor_1 == 'unspecified' and factor_2 == 'unspecified':
        return 'unspecified'
    
    # Rule 3: Use the stronger factor based on the predefined ranking
    if factor_strength_ranking[factor_1] < factor_strength_ranking[factor_2]:
        return factor_1
    else:
        return factor_2

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

category_strength_ranking = {
    'driver_related': 1,
    'non_driver_related': 2,
    'unspecified': 3  # Adding 'unspecified' with the lowest rank
}

# Create a new column to store the combined categories based on similar rules as before
def combine_categories(row):
    category_1 = row['factors_category_vehicle_1']
    category_2 = row['factors_category_vehicle_2']
    
    # Rule 1: If category_1 is not 'unspecified' and category_2 is 'unspecified', use category_1
    if category_1 != 'unspecified' and category_2 == 'unspecified':
        return category_1
    
    # Rule 2: If both are 'unspecified', then the new column should also show 'unspecified'
    if category_1 == 'unspecified' and category_2 == 'unspecified':
        return 'unspecified'
    
    # Rule 3: Use the stronger category based on the predefined ranking
    if category_strength_ranking[category_1] < category_strength_ranking[category_2]:
        return category_1
    else:
        return category_2

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def wrangle_coll_stage0(year, app_token, max_observations=None):
    df = get_data(year, app_token, max_observations)
    df = df.drop_duplicates(keep='first')
    df = date_time(df)
    df = initial_reorder_cols(df)
    df = drop_3plus(df)
    df = single_coll(df)
    df = motorist_only(df)
    df = second_drop(df)
    df = null_zero_loc(df)
    df = fill_geo(df)
    df = fill_na(df)
    df = drop_na(df)
    df = injuries_build(df)
    df = df.drop(columns=['location'])
    df['crash_datetime'] = pd.to_datetime(df['crash_datetime'])
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df['crash_time'] = pd.to_datetime(df['crash_time'], format='%H:%M:%S').dt.time
    df['zip_code'] = df['zip_code'].astype('int64')
    df = lower_case(df)
    # df = map_categories(df)
    # df = df.drop(columns=['vehicle_type_code1', 'contributing_factor_vehicle_1','vehicle_type_code2', 'contributing_factor_vehicle_2'])
    # df['borough'] = df['borough'].replace('the bronx', 'bronx')
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def add_injury_ratio(df):
    # Aggregate the data to get the count of injuries and crashes for each type of vehicle
    agg_data = df.groupby('vehicle_1_category')['injuries'].sum().reset_index()
    agg_data_crashes = df.groupby('vehicle_1_category').size().reset_index(name='count_of_crashes')
    
    # Merge the two aggregated dataframes
    merged_data = pd.merge(agg_data, agg_data_crashes, left_on='vehicle_1_category', right_on='vehicle_1_category', how='inner')
    
    # Calculate the ratio of crashes resulting in injuries by the number of crashes for each type of vehicle
    merged_data['injury_ratio'] = merged_data['injuries'] / merged_data['count_of_crashes']
    
    # Merge this ratio back into the original DataFrame
    df = pd.merge(df, merged_data[['vehicle_1_category', 'injury_ratio']], on='vehicle_1_category', how='left')
    
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def wrangle_coll_stage1(year, app_token, max_observations=None):
    df = get_data(year, app_token, max_observations)
    df = df.drop_duplicates(keep='first')
    df = date_time(df)
    df = initial_reorder_cols(df)
    df = drop_3plus(df)
    df = single_coll(df)
    df = motorist_only(df)
    df = second_drop(df)
    df = null_zero_loc(df)
    df = fill_geo(df)
    df = fill_na(df)
    df = drop_na(df)
    df = injuries_build(df)
    df = df.drop(columns=['location'])
    df['crash_datetime'] = pd.to_datetime(df['crash_datetime'])
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df['crash_time'] = pd.to_datetime(df['crash_time'], format='%H:%M:%S').dt.time
    df['zip_code'] = df['zip_code'].astype('int64')
    df = lower_case(df)
    df = map_categories(df)
    df = df.drop(columns=['vehicle_type_code1', 'contributing_factor_vehicle_1','vehicle_type_code2', 'contributing_factor_vehicle_2'])
    df['borough'] = df['borough'].replace('the bronx', 'bronx')
    
    # Apply the function to create the new combined column
    df['factors_subcat_vehicles'] = df.apply(combine_factors, axis=1)

    # Apply the function to create the new combined column for categories
    df['factors_category_vehicles'] = df.apply(combine_categories, axis=1)
    
    # Create the new 'vehicles' column by concatenating 'vehicle_1_category' twice with "_to_" in between
    df['vehicles'] = df['vehicle_1_category'] + "_to_" + df['vehicle_2_category']
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def wrangle_coll_stage2(year, app_token, max_observations=None):
    df = wrangle_coll_stage1(year, app_token, max_observations)
    df = datetime_features(df)
    return df

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def wrangle_coll_stage3(year, app_token, max_observations=None):
    df = wrangle_coll_stage2(year, app_token, max_observations)
    df['borough'].fillna('none', inplace=True)
    # df = add_injury_ratio(df)
    df = df.drop(columns=['vehicle_1_category', 'vehicle_2_category', 'factors_category_vehicle_1', 'factors_category_vehicle_2',	'factors_subcat_vehicle_1', 'factors_subcat_vehicle_2'])
    return df
