"""
Data loading and preprocessing module
"""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import config

class StockDataLoader:
    """Load and preprocess stock data"""
    
    def __init__(self, ticker: str, period: str = config.DATA_PERIOD):
        self.ticker = ticker
        self.period = period
        self.data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        print(f"📊 Fetching data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker: {self.ticker}")
        
        print(f"✅ Fetched {len(self.data)} trading days")
        return self.data
    
    def save_raw_data(self):
        """Save raw data to CSV"""
        filepath = config.RAW_DATA_DIR / f"{self.ticker}_{self.period}.csv"
        self.data.to_csv(filepath)
        print(f"💾 Saved raw data to {filepath}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        filepath = config.RAW_DATA_DIR / f"{self.ticker}_{self.period}.csv"
        if filepath.exists():
            self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"📂 Loaded raw data from {filepath}")
            return self.data
        else:
            raise FileNotFoundError(f"No saved data found for {self.ticker}")
    
    def get_latest_price(self) -> float:
        """Get the latest closing price"""
        return self.data['Close'].iloc[-1] if self.data is not None else None
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of the data"""
        if self.data is None:
            return {}
        
        return {
            'ticker': self.ticker,
            'start_date': self.data.index[0],
            'end_date': self.data.index[-1],
            'total_days': len(self.data),
            'latest_price': self.get_latest_price(),
            'mean_volume': self.data['Volume'].mean(),
            'price_range': (self.data['Close'].min(), self.data['Close'].max())
        }
