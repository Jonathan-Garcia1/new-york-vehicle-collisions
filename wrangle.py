import pandas as pd
import os
import requests


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
    # Convert 'crash_date' and 'crash_time' to datetime and combine them into a single column
    df['crash_datetime'] = pd.to_datetime(df['crash_date'].str[:10] + ' ' + df['crash_time'], format='%Y-%m-%d %H:%M')
    df['crash_date'] = pd.to_datetime(df['crash_date'])
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

def wrangle_coll_stage1(year, app_token, max_observations=None):
    df = get_data(year, app_token, max_observations)
    df = date_time(df)
    df = initial_reorder_cols(df)
    df = drop_3plus(df)
    df = single_coll(df)
    return df