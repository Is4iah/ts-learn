# Install the fredapi package using pip
# !pip install fredapi

from fredapi import Fred


def get_fred_data(series_id, observation_start, observation_end, api_key='0e929d96cfbf2133c9a346a74670f033'):
    """
    Retrieves FRED data using the provided parameters.

    Parameters:
    series_id (str): The ID of the FRED series you want to retrieve.
    observation_start (str): The start date of the data you want to retrieve.
    observation_end (str): The end date of the data you want to retrieve.
    api_key (str): Your FRED API key.

    Returns:
    pandas.DataFrame: The retrieved FRED data.
    """
    # Create a FRED API client
    fred = fredapi.Fred(api_key=api_key)

    # Retrieve the FRED data using the provided parameters
    data = fred.get_series(series_id, observation_start=observation_start, observation_end=observation_end)

    # Return the retrieved data
    return data
