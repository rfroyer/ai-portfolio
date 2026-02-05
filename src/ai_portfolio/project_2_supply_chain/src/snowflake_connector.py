import snowflake.connector
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class SnowflakeConnector:
    def __init__(self):
        try:
            self.conn = snowflake.connector.connect(
                user=os.getenv("SNOWFLAKE_USER"),
                password=os.getenv("SNOWFLAKE_PASSWORD"),
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
                database=os.getenv("SNOWFLAKE_DATABASE"),
                schema=os.getenv("SNOWFLAKE_SCHEMA")
            )
        except Exception as e:
            print(f"Error connecting to Snowflake: {e}")
            self.conn = None

    def execute_query(self, query):
        """Executes a query and returns the result as a pandas DataFrame."""
        if not self.conn:
            print("Cannot execute query, no connection available.")
            return None
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            df = cursor.fetch_pandas_all()
            return df
        except Exception as e:
            print(f"Error executing query \'{query}\': {e}")
            return None
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()

    def get_sales_by_product(self, product_id):
        """Fetches sales data for a product and formats it for the model."""
        query = f"SELECT SALE_DATE, QUANTITY FROM SALES WHERE PRODUCT_ID = {product_id} ORDER BY SALE_DATE;"
        sales_df = self.execute_query(query)
        
        if sales_df is None or sales_df.empty:
            return pd.DataFrame() # Return an empty DataFrame if no data

        # --- THE FIX IS HERE --- #
        # Rename the uppercase columns from the database to the lowercase
        # names that our forecasting model expects ('ds' and 'y').
        sales_df.rename(columns={
            'SALE_DATE': 'ds',
            'QUANTITY': 'y'
        }, inplace=True)
        # ---------------------- #

        return sales_df

    def close(self):
        if self.conn:
            self.conn.close()

