import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class HoltsModel(ExponentialSmoothing):
    def __init__(self, alpha=0.5, beta=0.5, df=None, trend: str = 'add', seasonal: str = 'add', seasonal_periods: int = 12):
        
        super().__init__(endog=df['value'], 
                         trend=trend,
                         seasonal=seasonal,
                         seasonal_periods=seasonal_periods)
        self.alpha = alpha
        self.beta = beta

    def fit(self, **kwargs):
        super().fit(smoothing_level=self.alpha, smoothing_slope=self.beta, **kwargs)

    def predict(self, steps=12, **kwargs):
        return super().forecast(steps=steps, **kwargs)
    
