"""
Main pipeline integrating all components
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import time

from .data_ingestion import CCTVDataLoader, WeatherDataLoader
from .preprocessing import ImagePreprocessor
from .detection_model import WaterloggingDetector
from .postprocessing import MaskPostprocessor, TemporalSmoother
from .forecasting_model import WaterloggingForecaster
from .fusion import PredictionFusion
from .visualization import WaterloggingVisualizer
from .export_logger import ResultsExporter, SystemLogger, PerformanceMonitor


class WaterloggingPipeline:
    """End-to-end waterlogging detection and forecasting pipeline"""
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Configuration module or dictionary
        """
        self.config = config
        
        # Initialize logger
        self.logger = SystemLogger(
            log_dir=str(config.LOGS_DIR),
            log_level=config.LOG_LEVEL
        )
        
        self.logger.log_info("Initializing Waterlogging Detection & Forecasting Pipeline")
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(
            target_size=config.IMAGE_SIZE,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            apply_clahe=True
        )
        
        self.detector = WaterloggingDetector(
            model_name=config.DETECTION_MODEL_NAME,
            encoder_name=config.ENCODER_NAME,
            encoder_weights=config.ENCODER_WEIGHTS,
            device=config.DEVICE,
            threshold=config.DETECTION_THRESHOLD
        )
        
        self.postprocessor = MaskPostprocessor(
            kernel_size=config.MORPH_KERNEL_SIZE,
            min_area=config.MIN_WATERLOG_AREA
        )
        
        self.temporal_smoother = TemporalSmoother(
            alpha=config.TEMPORAL_SMOOTHING_ALPHA
        )
        
        self.forecaster = WaterloggingForecaster(
            model_type=config.FORECAST_MODEL_TYPE,
            sequence_length=config.SEQUENCE_LENGTH,
            device=config.DEVICE
        )
        
        self.fusion = PredictionFusion(
            detection_weight=config.DETECTION_WEIGHT,
            forecast_weight=config.FORECAST_WEIGHT,
            temporal_smoothing_alpha=config.TEMPORAL_SMOOTHING_ALPHA
        )
        
        self.visualizer = WaterloggingVisualizer(
            overlay_alpha=config.OVERLAY_ALPHA,
            colormap=config.COLORMAP
        )
        
        self.exporter = ResultsExporter(
            output_dir=str(config.OUTPUT_DIR),
            export_format=config.EXPORT_FORMAT
        )
        
        self.performance_monitor = PerformanceMonitor()
        
        self.logger.log_model_info(
            "WaterloggingDetector",
            self.detector.get_model_info()
        )
    
    def process_single_frame(
        self,
        frame: np.ndarray,
        frame_metadata: Dict,
        weather_data: Dict,
        visualize: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input frame (BGR format)
            frame_metadata: Frame metadata
            weather_data: Weather data dictionary
            visualize: Create visualization
            
        Returns:
            Tuple of (visualization_image, results_dict)
        """
        frame_start_time = time.time()
        
        try:
            # 1. Preprocessing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor, processed_image = self.preprocessor.preprocess(frame_rgb)
            
            # 2. Detection
            prob_map, detection_metadata = self.detector.predict(
                image_tensor,
                return_probability=True
            )
            binary_mask = (prob_map > self.config.DETECTION_THRESHOLD).astype(np.uint8)
            
            # 3. Postprocessing
            cleaned_mask, postprocess_stats = self.postprocessor.postprocess(
                binary_mask,
                prob_map
            )
            
            # Update detection metadata with postprocessing stats
            detection_metadata.update(postprocess_stats)
            
            # 4. Temporal smoothing
            smoothed_mask, smoothed_prob = self.temporal_smoother.smooth(
                cleaned_mask,
                prob_map
            )
            
            # 5. Forecasting
            forecast_risk, forecast_metadata = self.forecaster.predict(
                detection_metadata,
                weather_data
            )
            
            # 6. Fusion
            fused_risk, enhanced_mask, fusion_metadata = self.fusion.fuse_predictions(
                smoothed_mask,
                detection_metadata,
                forecast_risk,
                forecast_metadata
            )
            
            # 7. Visualization
            if visualize:
                vis_image = self.visualizer.create_risk_visualization(
                    frame_rgb,
                    enhanced_mask,
                    fusion_metadata['risk_level'],
                    fused_risk,
                    fusion_metadata['detection_risk'],
                    fusion_metadata['forecast_risk'],
                    fusion_metadata
                )
            else:
                vis_image = frame_rgb
            
            # 8. Export results
            self.exporter.add_result(
                frame_metadata,
                detection_metadata,
                forecast_metadata,
                fusion_metadata
            )
            
            # Record performance
            frame_time = time.time() - frame_start_time
            self.performance_monitor.record_frame(frame_time)
            
            # Log
            self.logger.log_frame_processing(
                frame_metadata.get('frame_number', 0),
                detection_metadata,
                forecast_metadata,
                fusion_metadata
            )
            
            # Compile results
            results = {
                'frame_metadata': frame_metadata,
                'detection_metadata': detection_metadata,
                'forecast_metadata': forecast_metadata,
                'fusion_metadata': fusion_metadata,
                'processing_time': frame_time,
                'mask': enhanced_mask,
                'probability_map': smoothed_prob
            }
            
            return vis_image, results
            
        except Exception as e:
            self.logger.log_error(f"Error processing frame {frame_metadata.get('frame_number', 0)}", e)
            raise
    
    def process_video(
        self,
        video_path: str,
        weather_data_path: Optional[str] = None,
        output_video_path: Optional[str] = None,
        visualize: bool = True,
        save_frames: bool = False
    ) -> Dict:
        """
        Process entire video through pipeline
        
        Args:
            video_path: Path to video file or image directory
            weather_data_path: Path to weather data file
            output_video_path: Path to save output video
            visualize: Create visualizations
            save_frames: Save individual frames
            
        Returns:
            Summary statistics
        """
        self.logger.log_info(f"Starting video processing: {video_path}")
        self.performance_monitor.start()
        
        # Initialize data loaders
        cctv_loader = CCTVDataLoader(
            source_path=video_path,
            frame_skip=self.config.FRAME_SKIP
        )
        
        if weather_data_path:
            weather_loader = WeatherDataLoader(weather_data_path)
        else:
            self.logger.log_warning("No weather data provided, using dummy data")
            weather_loader = WeatherDataLoader()
        
        # Process frames
        vis_frames = []
        frame_count = 0
        
        for frame, frame_metadata in cctv_loader.load_frames():
            # Get weather data
            weather_data = weather_loader.get_weather_at_timestamp(
                frame_metadata['timestamp'],
                history_hours=self.config.WEATHER_HISTORY_HOURS
            )
            
            # Process frame
            vis_frame, results = self.process_single_frame(
                frame,
                frame_metadata,
                weather_data,
                visualize=visualize
            )
            
            if visualize:
                vis_frames.append(vis_frame)
            
            # Save individual frame if requested
            if save_frames:
                frame_path = self.config.OUTPUT_DIR / "frames" / f"frame_{frame_count:06d}.jpg"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                self.visualizer.save_visualization(vis_frame, str(frame_path))
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                self.logger.log_info(f"Processed {frame_count} frames")
        
        # Save output video
        if output_video_path and vis_frames:
            self.visualizer.create_video_from_frames(
                vis_frames,
                output_video_path,
                fps=self.config.VIDEO_FPS
            )
            self.logger.log_info(f"Output video saved to {output_video_path}")
        
        # Export results
        self.exporter.export_results()
        summary = self.exporter.export_summary()
        
        # Create temporal plot
        plot_path = self.config.OUTPUT_DIR / "temporal_plot.png"
        self.visualizer.create_temporal_plot(
            self.fusion.prediction_history,
            str(plot_path)
        )
        
        # Performance statistics
        perf_stats = self.performance_monitor.get_statistics()
        self.logger.log_performance_metrics(
            perf_stats['total_time'],
            perf_stats['avg_fps'],
            perf_stats['total_frames']
        )
        self.performance_monitor.print_statistics()
        
        # Log summary
        self.logger.log_summary(summary)
        
        self.logger.log_info("Video processing completed")
        
        return {
            'summary': summary,
            'performance': perf_stats,
            'frames_processed': frame_count
        }
    
    def load_detection_model(self, checkpoint_path: str):
        """Load pretrained detection model"""
        self.detector.load_checkpoint(checkpoint_path)
        self.logger.log_info(f"Loaded detection model from {checkpoint_path}")
    
    def load_forecasting_model(self, checkpoint_path: str, input_size: int):
        """Load pretrained forecasting model"""
        self.forecaster.load_model(checkpoint_path, input_size)
        self.logger.log_info(f"Loaded forecasting model from {checkpoint_path}")
    
    def reset(self):
        """Reset pipeline state"""
        self.temporal_smoother.reset()
        self.forecaster.reset_history()
        self.fusion.reset()
        self.exporter.clear_buffer()
        self.logger.log_info("Pipeline reset")
