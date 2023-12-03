import yfinance as yf
import pandas as pd

def get_fred_data(fred, series_id: str):
    """
    Fetches time series data from FRED (Federal Reserve Economic Data) for a given series ID.

    Args:
    - fred: FRED API object or client.
    - series_id (str): Identifier for the specific time series data on FRED.

    Returns:
    - pd.DataFrame: DataFrame containing time series data for the specified series ID.
    """
    data = fred.get_series(series_id)
    data = pd.DataFrame(data).reset_index()
    data.columns = ['date', series_id]
    data['date'] = pd.to_datetime(data['date'])
    data = data.ffill()
    return data



def get_yfinance_data(ticker: str, start_date: str, end_date: str):
    
    """
    Fetches historical price data using yfinance for a given ticker symbol within a specified date range.

    Args:
    - ticker (str): Ticker symbol of the asset (e.g., stock, commodity, etc.).
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: DataFrame containing historical price data for the specified ticker and date range.
    
    """
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Reset index and rename columns
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'date'}, inplace=True)
    
    # Ensure 'date' column is in datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    return data


code_dic = {'DGS10': 'bond', 'DCOILWTICO': 'oil', 'DCOILBRENTEU': 'oil_EU', 'GOLDAMGBD228NLBM': 'gold'}