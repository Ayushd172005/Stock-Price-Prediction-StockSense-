"""
Model training module
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import config
from src.model import StockPricePredictor

class ModelTrainer:
    """Handle model training pipeline"""
    
    def __init__(self, df, feature_cols=None):
        self.df = df
        self.feature_cols = feature_cols or config.FEATURE_COLUMNS
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_sequences(self, seq_length=config.SEQUENCE_LENGTH):
        """Prepare sequences for LSTM"""
        print(f"🔄 Preparing sequences (length={seq_length})...")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(self.df[self.feature_cols])
        
        X, y = [], []
        for i in range(seq_length, len(scaled_features)):
            X.append(scaled_features[i-seq_length:i])
            y.append(self.df['Target'].iloc[i])
        
        X, y = np.array(X), np.array(y)
        
        # Convert to classification (0: Down, 1: Sideways, 2: Up)
        y = y + 1
        
        print(f"✅ Created {len(X)} sequences")
        print(f"   Shape: X={X.shape}, y={y.shape}")
        print(f"   Class distribution: {np.bincount(y.astype(int))}")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into train/val/test sets"""
        # First split: train and temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=config.RANDOM_STATE
        )
        
        # Second split: val and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_STATE
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train, X_val, y_val, ticker: str):
        """Train the model"""
        print("🎯 Training model...")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        predictor = StockPricePredictor(input_shape)
        self.model = predictor.build_model()
        
        # Get callbacks
        callbacks = predictor.get_callbacks(ticker)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        predictor.save_model(ticker)
        
        return history
    
    def save_scaler(self, ticker: str):
        """Save the fitted scaler"""
        filepath = config.SCALERS_DIR / f"{ticker}_scaler.pkl"
        joblib.dump(self.scaler, filepath)
        print(f"💾 Scaler saved to {filepath}")
    
    def load_scaler(self, ticker: str):
        """Load a saved scaler"""
        filepath = config.SCALERS_DIR / f"{ticker}_scaler.pkl"
        self.scaler = joblib.load(filepath)
        print(f"📂 Scaler loaded from {filepath}")
        return self.scaler
