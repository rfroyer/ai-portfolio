import pandas as pd
import pmdarima as pm
import warnings

warnings.filterwarnings("ignore")

class DemandForecaster:
    def predict(self, sales_data):
        """Finds the best SARIMA model quickly and predicts the next 30 days."""
        
        if sales_data.empty or len(sales_data) < 14:
            return pd.DataFrame(columns=["ds", "yhat"])

        sales_data["ds"] = pd.to_datetime(sales_data["ds"])
        sales_data.set_index("ds", inplace=True)
        daily_sales = sales_data["y"].resample("D").sum().fillna(0)

        try:
            # --- HIGH-PERFORMANCE CONFIGURATION --- #
            # We are giving the model a strict time budget by limiting the
            # number of models it is allowed to test to just 10.
            # This provides the best balance between speed and accuracy.
            model = pm.auto_arima(
                daily_sales,
                start_p=1, start_q=1,
                test='adf',
                max_p=2, max_q=2, # Constrain the non-seasonal part
                m=7,              # Defines the weekly seasonal cycle
                d=None,           # Let the model find the right level of differencing
                seasonal=True,    # Enable seasonal forecasting
                start_P=0, D=1,
                trace=False,      # Don't print the search steps
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,    # Use a fast search algorithm
                n_jobs=1,         # Run in a single thread to prevent deadlocks
                n_fits=10         # <-- THE KEY PERFORMANCE OPTIMIZATION
            )
            # ------------------------------------ #

            n_periods = 30
            forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
            future_dates = pd.date_range(start=daily_sales.index.max() + pd.DateOffset(days=1), periods=n_periods, freq="D")

            forecast_df = pd.DataFrame({"ds": future_dates, "yhat": forecast})
            return forecast_df

        except Exception as e:
            print(f"An error occurred in the high-performance model: {e}")
            return pd.DataFrame(columns=["ds", "yhat"])

