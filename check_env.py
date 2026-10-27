import sys
try:
    import fastapi
    import uvicorn
    import xgboost
    import joblib
    print(f"FastAPI: {fastapi.__version__}")
    print(f"XGBoost: {xgboost.__version__}")
    print(f"Python: {sys.version}")
    print("All dependencies OK")
except ImportError as e:
    print(f"MISSING: {e}")
