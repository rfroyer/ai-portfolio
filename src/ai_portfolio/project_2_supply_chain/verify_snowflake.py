import os
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

print("Testing Snowflake connection...")
print(f"Account: {os.getenv('SNOWFLAKE_ACCOUNT')}")
print(f"User: {os.getenv('SNOWFLAKE_USER')}")

try:
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT")
    )
    print("✅ Snowflake connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Error connecting to Snowflake: {e}")
