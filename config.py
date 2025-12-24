"""
Configuration file for StockSense
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"
SCALERS_DIR = MODEL_DIR / "scalers"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, SCALERS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
SEQUENCE_LENGTH = 60
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Feature columns
FEATURE_COLUMNS = [
    'Close', 'Volume', 'Returns', 'SMA_5', 'SMA_10', 'SMA_20',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI',
    'BB_Width', 'Stoch_K', 'ATR', 'Volume_Ratio', 'Momentum'
]

# Stock tickers for dashboard
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']

# Prediction thresholds
UP_THRESHOLD = 0.002  # 0.2%
DOWN_THRESHOLD = -0.002  # -0.2%

# API settings
DATA_PERIOD = '2y'
DATA_INTERVAL = '1d'
