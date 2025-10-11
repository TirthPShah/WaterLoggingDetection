"""
Waterlogging forecasting model using weather data and detection history
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import pickle


class LSTMForecaster(nn.Module):
    """LSTM-based forecasting model"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM forecaster
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, sequence_length, input_size)
            
        Returns:
            Risk prediction (batch, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        return output


class WaterloggingForecaster:
    """Waterlogging risk forecasting system"""
    
    def __init__(
        self,
        model_type: str = "lstm",
        sequence_length: int = 10,
        device: str = "cuda"
    ):
        """
        Initialize forecasting model
        
        Args:
            model_type: Type of model (lstm, random_forest, gradient_boosting)
            sequence_length: Number of timesteps to use for prediction
            device: Device to run model on
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Feature history buffer
        self.feature_history = []
    
    def create_model(self, input_size: int):
        """
        Create forecasting model
        
        Args:
            input_size: Number of input features
        """
        if self.model_type == "lstm":
            self.model = LSTMForecaster(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_features(
        self,
        detection_output: Dict,
        weather_data: Dict
    ) -> np.ndarray:
        """
        Prepare feature vector from detection and weather data
        
        Args:
            detection_output: Detection model output with statistics
            weather_data: Weather data dictionary
            
        Returns:
            Feature vector
        """
        features = []
        
        # Detection features
        features.append(detection_output.get('waterlogged_ratio', 0.0))
        features.append(detection_output.get('mean_probability', 0.0))
        features.append(detection_output.get('max_probability', 0.0))
        features.append(detection_output.get('num_regions', 0))
        features.append(detection_output.get('largest_region_size', 0) / 1000.0)  # Normalize
        
        # Weather features
        features.append(weather_data.get('current_rainfall_mm', 0.0))
        features.append(weather_data.get('current_humidity_percent', 50.0) / 100.0)
        features.append(weather_data.get('current_temperature_c', 25.0) / 50.0)  # Normalize
        features.append(weather_data.get('avg_rainfall_6h', 0.0))
        features.append(weather_data.get('max_rainfall_6h', 0.0))
        features.append(weather_data.get('cumulative_rainfall_6h', 0.0))
        features.append(weather_data.get('avg_humidity_6h', 50.0) / 100.0)
        
        return np.array(features, dtype=np.float32)
    
    def update_history(self, features: np.ndarray):
        """
        Update feature history buffer
        
        Args:
            features: Feature vector for current timestep
        """
        self.feature_history.append(features)
        
        # Maintain buffer size
        if len(self.feature_history) > self.sequence_length:
            self.feature_history.pop(0)
    
    def predict(
        self,
        detection_output: Dict,
        weather_data: Dict,
        return_confidence: bool = False
    ) -> Tuple[float, Dict]:
        """
        Predict waterlogging risk
        
        Args:
            detection_output: Detection model output
            weather_data: Weather data
            return_confidence: Return confidence/uncertainty estimate
            
        Returns:
            Tuple of (risk_score, metadata)
            - risk_score: Predicted risk (0-1)
            - metadata: Additional prediction information
        """
        # Prepare features
        features = self.prepare_features(detection_output, weather_data)
        
        # Update history
        self.update_history(features)
        
        # Need enough history for prediction
        if len(self.feature_history) < self.sequence_length:
            # Not enough history, return current detection as risk
            risk_score = detection_output.get('waterlogged_ratio', 0.0)
            metadata = {
                'method': 'current_detection',
                'history_available': len(self.feature_history),
                'history_required': self.sequence_length
            }
            return risk_score, metadata
        
        # Prepare sequence
        sequence = np.array(self.feature_history[-self.sequence_length:])
        
        # Predict based on model type
        if self.model_type == "lstm":
            risk_score = self._predict_lstm(sequence)
        else:
            risk_score = self._predict_sklearn(sequence)
        
        metadata = {
            'method': self.model_type,
            'history_length': len(self.feature_history),
            'current_waterlogged_ratio': detection_output.get('waterlogged_ratio', 0.0),
            'current_rainfall': weather_data.get('current_rainfall_mm', 0.0)
        }
        
        return float(risk_score), metadata
    
    def _predict_lstm(self, sequence: np.ndarray) -> float:
        """
        LSTM prediction
        
        Args:
            sequence: Feature sequence (sequence_length, input_size)
            
        Returns:
            Risk score
        """
        if self.model is None:
            return 0.0
        
        self.model.eval()
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
        
        risk_score = prediction.item()
        
        return risk_score
    
    def _predict_sklearn(self, sequence: np.ndarray) -> float:
        """
        Sklearn model prediction
        
        Args:
            sequence: Feature sequence
            
        Returns:
            Risk score
        """
        if self.model is None or not self.is_fitted:
            return 0.0
        
        # Flatten sequence for sklearn models
        features_flat = sequence.flatten().reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_flat)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Clip to [0, 1]
        risk_score = np.clip(prediction, 0.0, 1.0)
        
        return risk_score
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train forecasting model
        
        Args:
            X_train: Training sequences (N, sequence_length, input_size)
            y_train: Training targets (N,)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs (for LSTM)
            batch_size: Batch size (for LSTM)
            learning_rate: Learning rate (for LSTM)
        """
        if self.model_type == "lstm":
            self._train_lstm(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate)
        else:
            self._train_sklearn(X_train, y_train)
        
        self.is_fitted = True
    
    def _train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ):
        """Train LSTM model"""
        # Create model if not exists
        if self.model is None:
            input_size = X_train.shape[2]
            self.create_model(input_size)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _train_sklearn(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train sklearn model"""
        # Create model if not exists
        if self.model is None:
            input_size = X_train.shape[1] * X_train.shape[2]
            self.create_model(input_size)
        
        # Flatten sequences
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Fit scaler
        self.scaler.fit(X_train_flat)
        X_train_scaled = self.scaler.transform(X_train_flat)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        print(f"{self.model_type} model trained successfully")
    
    def save_model(self, filepath: str):
        """Save model to file"""
        state = {
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'is_fitted': self.is_fitted,
        }
        
        if self.model_type == "lstm":
            state['model_state_dict'] = self.model.state_dict()
        else:
            state['model'] = self.model
            state['scaler'] = self.scaler
        
        if self.model_type == "lstm":
            torch.save(state, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, input_size: int = None):
        """Load model from file"""
        if self.model_type == "lstm":
            state = torch.load(filepath, map_location=self.device)
            
            if input_size is None:
                raise ValueError("input_size required for LSTM model")
            
            self.create_model(input_size)
            self.model.load_state_dict(state['model_state_dict'])
            self.model.eval()
        else:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.model = state['model']
            self.scaler = state['scaler']
        
        self.is_fitted = state['is_fitted']
        
        print(f"Model loaded from {filepath}")
    
    def reset_history(self):
        """Reset feature history buffer"""
        self.feature_history = []
