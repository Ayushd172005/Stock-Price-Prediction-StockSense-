"""
Feature engineering module for technical indicators
"""
import pandas as pd
import numpy as np
import ta
import config

class FeatureEngineering:
    """Create technical indicators and features"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def create_all_features(self) -> pd.DataFrame:
        """Create all technical indicators"""
        print("🔧 Creating technical indicators...")
        
        self._add_returns()
        self._add_moving_averages()
        self._add_macd()
        self._add_rsi()
        self._add_bollinger_bands()
        self._add_stochastic()
        self._add_atr()
        self._add_volume_indicators()
        self._add_momentum()
        self._create_target()
        
        # Drop NaN values
        initial_len = len(self.df)
        self.df.dropna(inplace=True)
        print(f"✅ Created features. Dropped {initial_len - len(self.df)} rows with NaN")
        
        return self.df
    
    def _add_returns(self):
        """Add return-based features"""
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
    
    def _add_moving_averages(self):
        """Add Simple and Exponential Moving Averages"""
        self.df['SMA_5'] = ta.trend.sma_indicator(self.df['Close'], window=5)
        self.df['SMA_10'] = ta.trend.sma_indicator(self.df['Close'], window=10)
        self.df['SMA_20'] = ta.trend.sma_indicator(self.df['Close'], window=20)
        self.df['EMA_12'] = ta.trend.ema_indicator(self.df['Close'], window=12)
        self.df['EMA_26'] = ta.trend.ema_indicator(self.df['Close'], window=26)
    
    def _add_macd(self):
        """Add MACD indicators"""
        macd = ta.trend.MACD(self.df['Close'])
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()
        self.df['MACD_Diff'] = macd.macd_diff()
    
    def _add_rsi(self):
        """Add Relative Strength Index"""
        self.df['RSI'] = ta.momentum.rsi(self.df['Close'], window=14)
    
    def _add_bollinger_bands(self):
        """Add Bollinger Bands"""
        bollinger = ta.volatility.BollingerBands(self.df['Close'])
        self.df['BB_High'] = bollinger.bollinger_hband()
        self.df['BB_Low'] = bollinger.bollinger_lband()
        self.df['BB_Mid'] = bollinger.bollinger_mavg()
        self.df['BB_Width'] = (self.df['BB_High'] - self.df['BB_Low']) / self.df['BB_Mid']
    
    def _add_stochastic(self):
        """Add Stochastic Oscillator"""
        stoch = ta.momentum.StochasticOscillator(
            self.df['High'], self.df['Low'], self.df['Close']
        )
        self.df['Stoch_K'] = stoch.stoch()
        self.df['Stoch_D'] = stoch.stoch_signal()
    
    def _add_atr(self):
        """Add Average True Range"""
        self.df['ATR'] = ta.volatility.average_true_range(
            self.df['High'], self.df['Low'], self.df['Close']
        )
    
    def _add_volume_indicators(self):
        """Add volume-based indicators"""
        self.df['Volume_SMA'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_SMA']
    
    def _add_momentum(self):
        """Add momentum indicators"""
        self.df['Momentum'] = self.df['Close'] - self.df['Close'].shift(4)
    
    def _create_target(self):
        """Create target variable for prediction"""
        self.df['Price_Tomorrow'] = self.df['Close'].shift(-1)
        self.df['Target'] = 0  # Sideways
        
        # Up movement (>0.2%)
        self.df.loc[
            self.df['Price_Tomorrow'] > self.df['Close'] * (1 + config.UP_THRESHOLD),
            'Target'
        ] = 1
        
        # Down movement (<-0.2%)
        self.df.loc[
            self.df['Price_Tomorrow'] < self.df['Close'] * (1 + config.DOWN_THRESHOLD),
            'Target'
        ] = -1
    
    def get_feature_columns(self) -> list:
        """Return list of feature columns"""
        return config.FEATURE_COLUMNS
    
    def save_processed_data(self, ticker: str):
        """Save processed data"""
        filepath = config.PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
        self.df.to_csv(filepath)
        print(f"💾 Saved processed data to {filepath}")
