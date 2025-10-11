"""
Results export and logging system
"""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class ResultsExporter:
    """Export detection and forecasting results"""
    
    def __init__(self, output_dir: str, export_format: str = "json"):
        """
        Initialize results exporter
        
        Args:
            output_dir: Directory to save results
            export_format: Export format (json, csv, both)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_format = export_format
        
        # Results buffer
        self.results_buffer = []
    
    def add_result(
        self,
        frame_metadata: Dict,
        detection_metadata: Dict,
        forecast_metadata: Dict,
        fusion_metadata: Dict
    ):
        """
        Add a result to the buffer
        
        Args:
            frame_metadata: Frame information
            detection_metadata: Detection results
            forecast_metadata: Forecast results
            fusion_metadata: Fusion results
        """
        result = {
            'frame': frame_metadata,
            'detection': detection_metadata,
            'forecast': forecast_metadata,
            'fusion': fusion_metadata,
            'export_timestamp': datetime.now().isoformat()
        }
        
        self.results_buffer.append(result)
    
    def export_results(self, filename: str = None):
        """
        Export buffered results to file
        
        Args:
            filename: Optional custom filename
        """
        if not self.results_buffer:
            print("No results to export")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}"
        
        # Export based on format
        if self.export_format in ["json", "both"]:
            self._export_json(filename)
        
        if self.export_format in ["csv", "both"]:
            self._export_csv(filename)
        
        print(f"Exported {len(self.results_buffer)} results")
    
    def _export_json(self, filename: str):
        """Export results as JSON"""
        output_path = self.output_dir / f"{filename}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results_buffer, f, indent=2, default=self._json_serializer)
        
        print(f"JSON results saved to {output_path}")
    
    def _export_csv(self, filename: str):
        """Export results as CSV (flattened)"""
        output_path = self.output_dir / f"{filename}.csv"
        
        if not self.results_buffer:
            return
        
        # Flatten nested dictionaries
        flattened_results = [self._flatten_dict(r) for r in self.results_buffer]
        
        # Get all possible field names
        fieldnames = set()
        for result in flattened_results:
            fieldnames.update(result.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_results)
        
        print(f"CSV results saved to {output_path}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to comma-separated strings
                if v and isinstance(v[0], (int, float, str)):
                    items.append((new_key, ','.join(map(str, v))))
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def export_summary(self, filename: str = None):
        """
        Export summary statistics
        
        Args:
            filename: Optional custom filename
        """
        if not self.results_buffer:
            print("No results to summarize")
            return
        
        # Calculate summary statistics
        fused_risks = [r['fusion']['fused_risk_score'] for r in self.results_buffer]
        detection_risks = [r['fusion']['detection_risk'] for r in self.results_buffer]
        forecast_risks = [r['fusion']['forecast_risk'] for r in self.results_buffer]
        
        summary = {
            'total_frames': len(self.results_buffer),
            'timestamp': datetime.now().isoformat(),
            'fused_risk': {
                'mean': float(np.mean(fused_risks)),
                'std': float(np.std(fused_risks)),
                'min': float(np.min(fused_risks)),
                'max': float(np.max(fused_risks)),
                'median': float(np.median(fused_risks))
            },
            'detection_risk': {
                'mean': float(np.mean(detection_risks)),
                'std': float(np.std(detection_risks)),
                'min': float(np.min(detection_risks)),
                'max': float(np.max(detection_risks))
            },
            'forecast_risk': {
                'mean': float(np.mean(forecast_risks)),
                'std': float(np.std(forecast_risks)),
                'min': float(np.min(forecast_risks)),
                'max': float(np.max(forecast_risks))
            },
            'risk_levels': {
                'low': sum(1 for r in self.results_buffer if r['fusion']['risk_level'] == 'low'),
                'medium': sum(1 for r in self.results_buffer if r['fusion']['risk_level'] == 'medium'),
                'high': sum(1 for r in self.results_buffer if r['fusion']['risk_level'] == 'high')
            }
        }
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}"
        
        output_path = self.output_dir / f"{filename}.json"
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {output_path}")
        
        return summary
    
    def clear_buffer(self):
        """Clear results buffer"""
        self.results_buffer = []


class SystemLogger:
    """System logging for detection and forecasting pipeline"""
    
    def __init__(
        self,
        log_dir: str,
        log_level: str = "INFO",
        log_to_console: bool = True
    ):
        """
        Initialize system logger
        
        Args:
            log_dir: Directory to save log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_to_console: Also log to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('WaterloggingDetection')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"system_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level))
            self.logger.addHandler(console_handler)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        if log_to_console:
            console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def log_frame_processing(
        self,
        frame_number: int,
        detection_metadata: Dict,
        forecast_metadata: Dict,
        fusion_metadata: Dict
    ):
        """Log frame processing results"""
        self.logger.info(
            f"Frame {frame_number}: "
            f"Detection={detection_metadata.get('waterlogged_ratio', 0):.3f}, "
            f"Forecast={forecast_metadata.get('forecast_risk', 0):.3f}, "
            f"Fused={fusion_metadata.get('fused_risk_score', 0):.3f}, "
            f"Risk={fusion_metadata.get('risk_level', 'unknown')}"
        )
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None):
        """Log error"""
        if exception:
            self.logger.error(f"{error_message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(error_message)
    
    def log_warning(self, warning_message: str):
        """Log warning"""
        self.logger.warning(warning_message)
    
    def log_info(self, info_message: str):
        """Log info"""
        self.logger.info(info_message)
    
    def log_debug(self, debug_message: str):
        """Log debug"""
        self.logger.debug(debug_message)
    
    def log_model_info(self, model_name: str, model_info: Dict):
        """Log model information"""
        self.logger.info(f"Model '{model_name}' info: {model_info}")
    
    def log_performance_metrics(
        self,
        processing_time: float,
        fps: float,
        frames_processed: int
    ):
        """Log performance metrics"""
        self.logger.info(
            f"Performance - Frames: {frames_processed}, "
            f"Time: {processing_time:.2f}s, FPS: {fps:.2f}"
        )
    
    def log_summary(self, summary: Dict):
        """Log summary statistics"""
        self.logger.info(f"Summary statistics: {json.dumps(summary, indent=2)}")


class PerformanceMonitor:
    """Monitor and log system performance"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.start_time = None
        self.frame_times = []
        self.total_frames = 0
    
    def start(self):
        """Start monitoring"""
        self.start_time = datetime.now()
        self.frame_times = []
        self.total_frames = 0
    
    def record_frame(self, processing_time: float):
        """
        Record frame processing time
        
        Args:
            processing_time: Time taken to process frame (seconds)
        """
        self.frame_times.append(processing_time)
        self.total_frames += 1
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.frame_times:
            return {
                'total_frames': 0,
                'total_time': 0,
                'avg_fps': 0,
                'avg_frame_time': 0
            }
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_frame_time = np.mean(self.frame_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'avg_frame_time': avg_frame_time,
            'min_frame_time': float(np.min(self.frame_times)),
            'max_frame_time': float(np.max(self.frame_times)),
            'std_frame_time': float(np.std(self.frame_times))
        }
    
    def print_statistics(self):
        """Print performance statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("PERFORMANCE STATISTICS")
        print("="*50)
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Total Time: {stats['total_time']:.2f}s")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Average Frame Time: {stats['avg_frame_time']*1000:.2f}ms")
        print(f"Min Frame Time: {stats['min_frame_time']*1000:.2f}ms")
        print(f"Max Frame Time: {stats['max_frame_time']*1000:.2f}ms")
        print("="*50 + "\n")
