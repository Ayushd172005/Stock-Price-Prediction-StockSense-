"""
Prediction module
"""
import numpy as np
import pandas as pd
from src.model import StockPricePredictor
import config

class Predictor:
    """Make predictions on new data"""
    
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
    
    def predict(self, df, seq_length=config.SEQUENCE_LENGTH):
        """Make prediction on the latest data"""
        if len(df) < seq_length:
            raise ValueError(f"Need at least {seq_length} data points")
        
        # Scale features
        scaled_features = self.scaler.transform(df[self.feature_cols])
        
        # Get last sequence
        X = scaled_features[-seq_length:].reshape(1, seq_length, len(self.feature_cols))
        
        # Predict
        prediction = self.model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        class_names = ['Down ↓', 'Sideways →', 'Up ↑']
        
        return {
            'prediction': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'down': prediction[0][0],
                'sideways': prediction[0][1],
                'up': prediction[0][2]
            }
        }
    
    def batch_predict(self, X):
        """Make predictions on batch data"""
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes, predictions
