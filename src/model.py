"""
LSTM Model architecture
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import config

class StockPricePredictor:
    """Bidirectional LSTM model for stock price prediction"""
    
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """Build the LSTM architecture"""
        print("🏗️ Building Bidirectional LSTM model...")
        
        self.model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=self.input_shape),
            Dropout(0.3),
            
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
        return self.model
    
    def get_callbacks(self, model_name: str):
        """Get training callbacks"""
        checkpoint_path = config.SAVED_MODELS_DIR / f"{model_name}_best.h5"
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def save_model(self, model_name: str):
        """Save the trained model"""
        filepath = config.SAVED_MODELS_DIR / f"{model_name}_final.h5"
        self.model.save(filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, model_name: str):
        """Load a saved model"""
        filepath = config.SAVED_MODELS_DIR / f"{model_name}_final.h5"
        self.model = tf.keras.models.load_model(filepath)
        print(f"📂 Model loaded from {filepath}")
        return self.model
