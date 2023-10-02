import pandas as pd
import os
import requests


def get_data(year, app_token, max_observations=None):
    # Define the base API URL
    base_url = 'https://data.cityofnewyork.us/resource/h9gi-nx95.json'

    # Check if a CSV file with the specified year already exists
    csv_filename = f'ny_collision{year}.csv'
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
        url = f'{base_url}?$where=crash_date between "{year}-01-01T00:00:00.000" and "{year}-12-31T23:59:59.999"&$$app_token={app_token}&$offset={offset}&$limit={actual_page_size}'

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

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

    print(f"Data for {year} retrieved and saved to {csv_filename}.")

    return df