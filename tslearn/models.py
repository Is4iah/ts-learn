import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt


from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# from sklearn.metrics import mean_squared_error



@dataclass
class YFinanceDataProcessor:
    """Handles fetching and returning data. """
    ticker_symbol: str
    start_date: str
    end_date: str
    
    def __post_init__(self):
        """Operations that are performed after the initialization step."""
        self.data = yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date)
        
    def filterby(self, col_name):
        """Returns a filtered dataframe."""
        return self.data[[col_name]]

# Define the abstract base class
@dataclass
class Model(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.
    
    Methods decorated with @abstractmethod must be implemented; if not, the interpreter will throw
    an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """
    data: pd.DataFrame

    @abstractmethod
    def augment_data(self):
        pass

    @abstractmethod
    def predict(self) -> np.array:
        pass
    
    @abstractmethod
    def __name__(self):
        pass
    
    def plot1D(self, col_name, ticker):
        # Plot the time series
        plt.plot(self.data.index.values, self.data[col_name])

        # Add title and labels to the plot
        plt.title(f"Univariate Time Series {ticker}")
        plt.xlabel("Time")
        plt.ylabel("Price")

        # Display the plot
        plt.show()
    
class RNN(Model):
    def __name__(self):
        return "RNN"
    
    def augment_data(self):
        return "RNN Augment"

    def predict(self):
        return "RNN Predict"
    
class LSTM(Model):
    def __name__(self):
        return "LSTM"
    
    def augment_data(self):
        return "LSTM Augment"

    def predict(self):
        return "LSTM Predict"    

class MLP(Model):
    def __name__(self):
        return "MLP"
    
    def augment_data(self):
        return "MLP Augment"

    def _autolag_predict(self) -> np.array:
        pass
    
    def _newssource_predict(self) -> np.array:
        pass
    
    def predict(self, typeof="newssource") -> np.array:
        prediction_mapper = {
            "newssource": self._newsources_predict,
            "autolag": self._autolag_predict,
        }
        self.cur_typeof_prediction = typeof
        
        return prediction_mapper[typeof]()

class CNN(Model):
    def __name__(self):
        return "CNN"
    
    def augment_data(self):
        return "CNN Augment"

    def predict(self):
        return "CNN Predict"

class ARIMA(Model):
    def __name__(self):
        return "ARIMA"
    
    def augment_data(self):
        return "ARIMA Augment"

    def predict(self):
        return "ARIMA Predict"

@dataclass
class EvaluationMetric:
    """Investigate the philosphy/design behind typing in python. 
    
    https://realpython.com/python-type-checking/
    """
    models: List[Model]
    
    def __post_init__(self):
        pass
        
    def _get_eval_handles(self):
        """dir(self) grabs all of the available methods and attributes on the class itself."""
        function_names = [name for name in dir(self) if callable(getattr(self, name)) and not name.startswith("__")]
        # Filter for metrics that follow our specific specification and return their names
        function_names = [fn for fn in function_names if fn[:5] == "eval_"]
        # Get a handle on the function objects themselves
        functions = [getattr(self, name) for name in function_names]
        return functions
    
    def plot_forecast(self):
        """Plots the forecast of each model respectively on the same plot."""
        pass
    
    def perform_evaluations(self):
        eval_metrics = self._get_eval_handles()
        eval_results = {}
        
        # eval_metrics = [eval_mse, eval_mape, ...] --> these are actual function handles we can
        # iterate over
        for eval_metric in eval_metrics:
            eval_results[eval_metric.__name__] = eval_metric() # eval_metric() == self.eval_mse()
            
        # Update the state of our object
        self.eval_results = eval_results
        
    def eval_mse(self) -> float:
        # sklearn.metrics.mean_squared_error
        return "MSE"
    
    def eval_mse_personal(self) -> float:
        return "MSE_PERSONAL"
    
    def eval_mape(self) -> float:
        return "MAPE"
    
    def eval_mase(self) -> float:
        return "MASE"
    
    def eval_darian_metric(self):
        return "DARIAN"
    
    def eval_detravious_metric(self):
        return "DETRAVIOUS"
    
    def eval_USC(self):
        return "USC"
    
    def eval_CORNELL(self):
        return "CORNELL"
        
@dataclass
class EvaluationResults:
    """Our diagnostic or reporter class to be further implemented if desired."""
    eval_metric: EvaluationMetric
    
    def summarize(self, eval_metric):
        for model in self.eval_metric.models:
            print(f"Summary of Experiments from Model<{model.__name__()}>")
            print(f"Metrics:\n{eval_metric.eval_results}\n")
            
@dataclass
class UniTTs:
    data_processor: YFinanceDataProcessor
    models: List[Model]
    eval_metric: EvaluationMetric
    eval_results: EvaluationResults
    
    def __post_init__(self):
        pass
    
    def execute(self):
        print("Executing...")


class SES(ExponentialSmoothing):
    
    def __init__(self, df, y, trend=None, damped=False, seasonal=None, seasonal_periods=None, initialization_method="estimated", initial_level=None, initial_trend=None, use_boxcox=False, bounds=None, dates=None, freq=None, missing="none"):
        super().__init__(endog=y, trend=trend, damped=damped, seasonal=seasonal, seasonal_periods=seasonal_periods, initialization_method=initialization_method, initial_level=initial_level, initial_trend=initial_trend, use_boxcox=use_boxcox, bounds=bounds, dates=dates, freq=freq, missing=missing)
        
        self.df = df
        
        
        
    def __str__(self):
        return 'SES'
    
    def fit(smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None, damping_trend=None, optimized=True, remove_bias=False, start_params=None, method=None, minimize_kwargs=None, use_brute=True, use_boxcox=None, use_basinhopping=None, initial_level=None, initial_trend=None):
        """
        Fit the SES model to the data.

        Parameters:
        - optimized (bool): Whether or not to optimize the model parameters.
        - use_brute (bool): Whether to use brute force optimization.
        - remove_bias (bool): Whether to remove bias from the model.
        - use_basinhopping (bool): Whether to use basin-hopping optimization.

        Returns:
        - Fit object: A fitted SES model.
        """
        return super().fit(smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None, damping_trend=None, optimized=True, remove_bias=False, start_params=None, method=None, minimize_kwargs=None, use_brute=True, use_boxcox=None, use_basinhopping=None, initial_level=None, initial_trend=None)
    
    def forecast(self, steps=7):
        """
        Forecast future values.

        Parameters:
        - periods (int): The number of periods to forecast.

        Returns:
        - DataFrame: A DataFrame with 'report_date' and 'forecasted_value' columns.
        """
        
        # Fit the model
        fit = self.fit()

        # Make predictions for the specified number of periods
        forecast_values = fit.forecast(steps=steps)

        # Convert forecasted values to a DataFrame and reset the index
        forecast_values_df = pd.DataFrame(forecast_values).reset_index()

        # Rename columns and add the 'report_date' column
        forecast_values_df.columns = ['index', 'val']
        forecast_values_df['report_date'] = pd.date_range(start=self.df['report_date'].max(), periods=len(forecast_values_df), freq='W')
        
        # Reorder columns to have 'report_date' and 'val'
        forecast_values_df = forecast_values_df[['report_date', 'val']]

        return forecast_values_df

    
    def get_df(self):
        return self.df