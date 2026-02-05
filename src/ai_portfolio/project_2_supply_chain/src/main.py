from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio

from snowflake_connector import SnowflakeConnector
from train_model import DemandForecaster

app = FastAPI(title="Supply Chain API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
 )

try:
    db_connector = SnowflakeConnector()
    model = DemandForecaster()
except Exception as e:
    db_connector = None
    model = None

class Product(BaseModel):
    product_id: int
    product_name: str

class ProductList(BaseModel):
    products: List[Product]

class Forecast(BaseModel):
    ds: str
    yhat: float

class ForecastResponse(BaseModel):
    product_id: int
    forecast: List[Forecast]

class ProductSummary(BaseModel):
    total_revenue: float
    total_transactions: int
    total_inventory: int

@app.get("/products", response_model=ProductList)
def get_products():
    if not db_connector: raise HTTPException(500, "DB not connected")
    df = db_connector.execute_query("SELECT PRODUCT_ID, PRODUCT_NAME FROM PRODUCTS ORDER BY PRODUCT_ID;")
    if df is None: return {"products": []}
    df.columns = df.columns.str.lower()
    return {"products": df.to_dict("records")}

@app.get("/product-summary/{product_id}", response_model=ProductSummary)
def get_product_summary(product_id: int):
    if not db_connector: raise HTTPException(500, "DB not connected")
    try:
        rev_q = f"SELECT SUM(REVENUE) AS TOTAL_REVENUE FROM SALES WHERE PRODUCT_ID = {product_id};"
        trans_q = f"SELECT COUNT(SALE_ID) AS TOTAL_TRANSACTIONS FROM SALES WHERE PRODUCT_ID = {product_id};"
        inv_q = f"SELECT SUM(QUANTITY_ON_HAND) AS TOTAL_INVENTORY FROM INVENTORY WHERE PRODUCT_ID = {product_id};"
        total_revenue = db_connector.execute_query(rev_q).iloc[0,0] or 0
        total_transactions = db_connector.execute_query(trans_q).iloc[0,0] or 0
        total_inventory = db_connector.execute_query(inv_q).iloc[0,0] or 0
        return {
            "total_revenue": float(total_revenue),
            "total_transactions": int(total_transactions),
            "total_inventory": int(total_inventory),
        }
    except Exception: return {"total_revenue": 0, "total_transactions": 0, "total_inventory": 0}

@app.get("/forecast/{product_id}", response_model=ForecastResponse)
async def get_forecast(product_id: int):
    if not db_connector or not model: raise HTTPException(500, "Services not available")
    try:
        loop = asyncio.get_running_loop()
        sales_df = await loop.run_in_executor(None, db_connector.get_sales_by_product, product_id)
        if sales_df.empty:
            return {"product_id": product_id, "forecast": []}
        forecast_df = await loop.run_in_executor(None, model.predict, sales_df)
        forecast_records = forecast_df[["ds", "yhat"]].to_dict("records")
        for record in forecast_records:
            record["ds"] = record["ds"].isoformat()
        return {"product_id": product_id, "forecast": forecast_records}
    except Exception as e:
        return {"product_id": product_id, "forecast": []}
