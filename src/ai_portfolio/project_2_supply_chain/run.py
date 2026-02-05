import uvicorn
import sys
import os

# --- FIX: Add the 'src' directory to the Python path ---
# This allows the application to find the modules located in the src folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# -----------------------------------------------------

if __name__ == "__main__":
    # The app is now found at 'main:app' because 'src' is in the path
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True, 
        reload_dirs=[os.path.join(os.path.dirname(__file__), 'src')]
    )

