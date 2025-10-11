"""
Fusion module to integrate detection and forecasting predictions
"""

import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class PredictionFusion:
    """Fuse detection and forecasting predictions"""
    
    def __init__(
        self,
        detection_weight: float = 0.6,
        forecast_weight: float = 0.4,
        temporal_smoothing_alpha: float = 0.3
    ):
        """
        Initialize prediction fusion
        
        Args:
            detection_weight: Weight for detection model (0-1)
            forecast_weight: Weight for forecasting model (0-1)
            temporal_smoothing_alpha: Exponential smoothing factor
        """
        assert abs(detection_weight + forecast_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        
        self.detection_weight = detection_weight
        self.forecast_weight = forecast_weight
        self.temporal_smoothing_alpha = temporal_smoothing_alpha
        
        # History for temporal smoothing
        self.previous_risk_score = None
        self.prediction_history = []
    
    def fuse_predictions(
        self,
        detection_mask: np.ndarray,
        detection_metadata: Dict,
        forecast_risk: float,
        forecast_metadata: Dict,
        apply_temporal_smoothing: bool = True
    ) -> Tuple[float, np.ndarray, Dict]:
        """
        Fuse detection and forecasting predictions
        
        Args:
            detection_mask: Binary mask from detection model
            detection_metadata: Detection statistics
            forecast_risk: Forecasted risk score (0-1)
            forecast_metadata: Forecast metadata
            apply_temporal_smoothing: Apply temporal smoothing
            
        Returns:
            Tuple of (fused_risk_score, enhanced_mask, fusion_metadata)
        """
        # Extract detection risk
        detection_risk = detection_metadata.get('waterlogged_ratio', 0.0)
        
        # Calculate weighted fusion
        fused_risk = (
            self.detection_weight * detection_risk +
            self.forecast_weight * forecast_risk
        )
        
        # Apply temporal smoothing
        if apply_temporal_smoothing and self.previous_risk_score is not None:
            fused_risk = (
                self.temporal_smoothing_alpha * fused_risk +
                (1 - self.temporal_smoothing_alpha) * self.previous_risk_score
            )
        
        # Update history
        self.previous_risk_score = fused_risk
        
        # Enhance mask with forecast information
        enhanced_mask = self._enhance_mask_with_forecast(
            detection_mask,
            forecast_risk,
            detection_risk
        )
        
        # Create fusion metadata
        fusion_metadata = {
            'fused_risk_score': float(fused_risk),
            'detection_risk': float(detection_risk),
            'forecast_risk': float(forecast_risk),
            'detection_weight': self.detection_weight,
            'forecast_weight': self.forecast_weight,
            'temporal_smoothing_applied': apply_temporal_smoothing,
            'risk_level': self._classify_risk_level(fused_risk),
            'timestamp': datetime.now().isoformat(),
            'detection_metadata': detection_metadata,
            'forecast_metadata': forecast_metadata
        }
        
        # Add to prediction history
        self.prediction_history.append({
            'timestamp': fusion_metadata['timestamp'],
            'fused_risk': fused_risk,
            'detection_risk': detection_risk,
            'forecast_risk': forecast_risk
        })
        
        return fused_risk, enhanced_mask, fusion_metadata
    
    def _enhance_mask_with_forecast(
        self,
        detection_mask: np.ndarray,
        forecast_risk: float,
        detection_risk: float
    ) -> np.ndarray:
        """
        Enhance detection mask using forecast information
        
        Args:
            detection_mask: Binary mask from detection
            forecast_risk: Forecasted risk
            detection_risk: Current detection risk
            
        Returns:
            Enhanced mask
        """
        enhanced_mask = detection_mask.copy()
        
        # If forecast predicts higher risk than current detection,
        # we might want to dilate the mask slightly
        risk_difference = forecast_risk - detection_risk
        
        if risk_difference > 0.2:  # Forecast significantly higher
            # Dilate mask to indicate potential spread
            import cv2
            kernel_size = int(3 + risk_difference * 10)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size)
            )
            enhanced_mask = cv2.dilate(enhanced_mask, kernel, iterations=1)
        
        return enhanced_mask
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """
        Classify risk level
        
        Args:
            risk_score: Risk score (0-1)
            
        Returns:
            Risk level string
        """
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        else:
            return "high"
    
    def get_risk_trend(self, window_size: int = 10) -> Dict:
        """
        Analyze risk trend over recent predictions
        
        Args:
            window_size: Number of recent predictions to analyze
            
        Returns:
            Trend analysis dictionary
        """
        if len(self.prediction_history) < 2:
            return {
                'trend': 'unknown',
                'change_rate': 0.0,
                'samples': len(self.prediction_history)
            }
        
        # Get recent history
        recent_history = self.prediction_history[-window_size:]
        
        # Extract fused risks
        risks = [h['fused_risk'] for h in recent_history]
        
        # Calculate trend
        if len(risks) >= 2:
            # Simple linear regression
            x = np.arange(len(risks))
            y = np.array(risks)
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Classify trend
            if abs(slope) < 0.01:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            change_rate = float(slope)
        else:
            trend = 'unknown'
            change_rate = 0.0
        
        return {
            'trend': trend,
            'change_rate': change_rate,
            'samples': len(recent_history),
            'current_risk': risks[-1] if risks else 0.0,
            'average_risk': float(np.mean(risks)) if risks else 0.0,
            'max_risk': float(np.max(risks)) if risks else 0.0,
            'min_risk': float(np.min(risks)) if risks else 0.0
        }
    
    def update_weights(self, detection_weight: float, forecast_weight: float):
        """
        Update fusion weights
        
        Args:
            detection_weight: New detection weight
            forecast_weight: New forecast weight
        """
        assert abs(detection_weight + forecast_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        
        self.detection_weight = detection_weight
        self.forecast_weight = forecast_weight
    
    def reset(self):
        """Reset fusion state"""
        self.previous_risk_score = None
        self.prediction_history = []


class AdaptiveFusion:
    """Adaptive fusion that adjusts weights based on performance"""
    
    def __init__(
        self,
        initial_detection_weight: float = 0.6,
        initial_forecast_weight: float = 0.4,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize adaptive fusion
        
        Args:
            initial_detection_weight: Initial detection weight
            initial_forecast_weight: Initial forecast weight
            adaptation_rate: Rate of weight adaptation
        """
        self.detection_weight = initial_detection_weight
        self.forecast_weight = initial_forecast_weight
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.detection_errors = []
        self.forecast_errors = []
    
    def fuse_predictions(
        self,
        detection_risk: float,
        forecast_risk: float,
        ground_truth: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Adaptive fusion with optional ground truth for weight update
        
        Args:
            detection_risk: Detection model risk
            forecast_risk: Forecast model risk
            ground_truth: Optional ground truth for adaptation
            
        Returns:
            Tuple of (fused_risk, metadata)
        """
        # Fuse predictions
        fused_risk = (
            self.detection_weight * detection_risk +
            self.forecast_weight * forecast_risk
        )
        
        # Update weights if ground truth is available
        if ground_truth is not None:
            self._update_weights(detection_risk, forecast_risk, ground_truth)
        
        metadata = {
            'fused_risk': float(fused_risk),
            'detection_risk': float(detection_risk),
            'forecast_risk': float(forecast_risk),
            'detection_weight': float(self.detection_weight),
            'forecast_weight': float(self.forecast_weight)
        }
        
        return fused_risk, metadata
    
    def _update_weights(
        self,
        detection_risk: float,
        forecast_risk: float,
        ground_truth: float
    ):
        """
        Update weights based on prediction errors
        
        Args:
            detection_risk: Detection prediction
            forecast_risk: Forecast prediction
            ground_truth: Ground truth value
        """
        # Calculate errors
        detection_error = abs(detection_risk - ground_truth)
        forecast_error = abs(forecast_risk - ground_truth)
        
        # Store errors
        self.detection_errors.append(detection_error)
        self.forecast_errors.append(forecast_error)
        
        # Calculate average errors (recent window)
        window_size = 20
        recent_detection_errors = self.detection_errors[-window_size:]
        recent_forecast_errors = self.forecast_errors[-window_size:]
        
        avg_detection_error = np.mean(recent_detection_errors)
        avg_forecast_error = np.mean(recent_forecast_errors)
        
        # Adjust weights inversely proportional to errors
        total_error = avg_detection_error + avg_forecast_error
        
        if total_error > 0:
            # Better model gets higher weight
            new_detection_weight = (1 - avg_detection_error / total_error)
            new_forecast_weight = (1 - avg_forecast_error / total_error)
            
            # Normalize
            total_weight = new_detection_weight + new_forecast_weight
            new_detection_weight /= total_weight
            new_forecast_weight /= total_weight
            
            # Smooth update with adaptation rate
            self.detection_weight = (
                (1 - self.adaptation_rate) * self.detection_weight +
                self.adaptation_rate * new_detection_weight
            )
            self.forecast_weight = (
                (1 - self.adaptation_rate) * self.forecast_weight +
                self.adaptation_rate * new_forecast_weight
            )
