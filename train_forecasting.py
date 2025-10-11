"""
Training script for waterlogging forecasting model
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import config
from src.forecasting_model import WaterloggingForecaster


def prepare_training_data(
    detection_results_path: str,
    weather_data_path: str,
    sequence_length: int = 10
):
    """
    Prepare training data from historical detection results and weather data
    
    Args:
        detection_results_path: Path to detection results JSON/CSV
        weather_data_path: Path to weather data
        sequence_length: Sequence length for LSTM
        
    Returns:
        Tuple of (X, y) - features and targets
    """
    print("Preparing training data...")
    
    # Load detection results
    if detection_results_path.endswith('.csv'):
        detection_df = pd.read_csv(detection_results_path)
    else:
        detection_df = pd.read_json(detection_results_path)
    
    # Load weather data
    if weather_data_path.endswith('.csv'):
        weather_df = pd.read_csv(weather_data_path)
    else:
        weather_df = pd.read_json(weather_data_path)
    
    # Merge on timestamp
    detection_df['timestamp'] = pd.to_datetime(detection_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    merged_df = pd.merge_asof(
        detection_df.sort_values('timestamp'),
        weather_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )
    
    # Create feature vectors
    features = []
    targets = []
    
    for i in range(len(merged_df) - sequence_length - 1):
        # Extract sequence
        seq_data = merged_df.iloc[i:i+sequence_length]
        
        # Create feature vector for each timestep
        seq_features = []
        for _, row in seq_data.iterrows():
            feature_vec = [
                row.get('waterlogged_ratio', 0.0),
                row.get('mean_probability', 0.0),
                row.get('max_probability', 0.0),
                row.get('num_regions', 0),
                row.get('largest_region_size', 0) / 1000.0,
                row.get('rainfall_mm', 0.0),
                row.get('humidity_percent', 50.0) / 100.0,
                row.get('temperature_c', 25.0) / 50.0,
                row.get('avg_rainfall_6h', 0.0),
                row.get('max_rainfall_6h', 0.0),
                row.get('cumulative_rainfall_6h', 0.0),
                row.get('avg_humidity_6h', 50.0) / 100.0,
            ]
            seq_features.append(feature_vec)
        
        # Target: waterlogged ratio at next timestep
        target_row = merged_df.iloc[i + sequence_length]
        target = target_row.get('waterlogged_ratio', 0.0)
        
        features.append(seq_features)
        targets.append(target)
    
    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    
    print(f"Created {len(X)} training samples")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def train_forecasting_model(
    detection_results_path: str,
    weather_data_path: str,
    output_dir: str,
    model_type: str = "lstm",
    sequence_length: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    test_size: float = 0.2
):
    """
    Train forecasting model
    
    Args:
        detection_results_path: Path to historical detection results
        weather_data_path: Path to weather data
        output_dir: Output directory for model
        model_type: Model type (lstm, random_forest, gradient_boosting)
        sequence_length: Sequence length for temporal models
        epochs: Number of epochs (for LSTM)
        batch_size: Batch size (for LSTM)
        learning_rate: Learning rate (for LSTM)
        test_size: Test set ratio
    """
    print("="*60)
    print("WATERLOGGING FORECASTING MODEL TRAINING")
    print("="*60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    X, y = prepare_training_data(
        detection_results_path,
        weather_data_path,
        sequence_length
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize forecaster
    print(f"\nInitializing {model_type} forecaster...")
    forecaster = WaterloggingForecaster(
        model_type=model_type,
        sequence_length=sequence_length,
        device=config.DEVICE
    )
    
    # Create model
    input_size = X.shape[2]
    forecaster.create_model(input_size)
    
    # Train model
    print("\nStarting training...")
    print("="*60)
    
    forecaster.train(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Evaluate
    print("\nEvaluating model...")
    if model_type == "lstm":
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import torch
        
        # Predictions
        forecaster.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(config.DEVICE)
        with torch.no_grad():
            y_pred = forecaster.model(X_test_tensor).cpu().numpy().flatten()
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nTest Metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    # Save model
    model_path = output_dir / f"{model_type}_forecaster.pth"
    forecaster.save_model(str(model_path))
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Model saved to: {model_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train Waterlogging Forecasting Model")
    
    parser.add_argument(
        "--detection-results",
        required=True,
        help="Path to historical detection results (CSV/JSON)"
    )
    parser.add_argument(
        "--weather-data",
        required=True,
        help="Path to weather data (CSV/JSON)"
    )
    parser.add_argument(
        "--output",
        default="models/forecasting",
        help="Output directory"
    )
    parser.add_argument(
        "--model-type",
        default="lstm",
        choices=["lstm", "random_forest", "gradient_boosting"],
        help="Model type"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (for LSTM)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (for LSTM)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (for LSTM)"
    )
    
    args = parser.parse_args()
    
    train_forecasting_model(
        detection_results_path=args.detection_results,
        weather_data_path=args.weather_data,
        output_dir=args.output,
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
