import numpy as np
import pandas
import matplotlib.pyplot as plt

from typing import TypeAlias, Any, Tuple, Union, Sequence, Optional
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

Array: TypeAlias = np.ndarray
Order: TypeAlias = Tuple[int, int, int]
Times: TypeAlias = Union[Sequence[float], Array, pandas.Series]

def FullTest(series: Times, alpha: float = 0.05) -> Tuple[bool, float]:
  """
  Augmented Dickey Fuller Stationary Test
  """
  adf_value, p_value, lags, observations, critical_values, ic_best = adfuller(series)
  stationary = adf_value < alpha
  return bool(stationary), float(p_value)

def ModelArima(train: Array, tester: int, order: Order, **kwargs: Any) -> Tuple[Array, float]:
  """
  Fit an ARIMA Model With Specific Order & Optional Parameters
  """
  model = ARIMA(train, order=order, **kwargs).fit()
  logits = model.forecast(tester)
  return logits, model.aic

def Helplot(train: Array, test: Array, logits: Array) -> None:
  """
  Helper Function to Plot Train & Test Result
  """
  fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))
  # Overall Plot
  axes[0].plot(train, label="Train")
  axes[0].plot(test, label="Test")
  axes[0].plot(logits, label="Result")
  axes[0].set_title("Overall Result on Forecasting")
  axes[0].legend()
  # Detail Plot
  axes[1].plot(test, label="Test")
  axes[1].plot(logits, label="Result")
  axes[1].set_title("Detail Result on Forecasting")
  axes[1].legend()
  plt.tight_layout();