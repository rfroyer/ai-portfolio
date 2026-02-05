import pandas as pd

class DataProcessor:
    def process_sales_data(self, data):
        df = pd.DataFrame(data, columns=["SALE_ID", "PRODUCT_ID", "SALE_DATE", "QUANTITY", "REVENUE"])
        df["SALE_DATE"] = pd.to_datetime(df["SALE_DATE"])
        return df
