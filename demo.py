"""
Demo script for AI-CCTV Waterlogging Detection & Forecasting System
"""

import argparse
from pathlib import Path
import config
from src.pipeline import WaterloggingPipeline
from src.data_ingestion import WeatherDataLoader

def run_demo_video(
    video_path: str,
    weather_data_path: str = None,
    output_dir: str = None,
    detection_checkpoint: str = None,
    forecast_checkpoint: str = None
):
    """
    Run demo on video or image directory
    
    Args:
        video_path: Path to video file or image directory
        weather_data_path: Path to weather data CSV/JSON (optional)
        output_dir: Output directory for results
        detection_checkpoint: Path to detection model checkpoint (optional)
        forecast_checkpoint: Path to forecasting model checkpoint (optional)
    """
    print("="*60)
    print("AI-CCTV WATERLOGGING DETECTION & FORECASTING SYSTEM")
    print("="*60)
    print(f"\nInput: {video_path}")
    print(f"Weather Data: {weather_data_path if weather_data_path else 'Using dummy data'}")
    
    # Update output directory if specified
    if output_dir:
        config.OUTPUT_DIR = Path(output_dir)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    print("\n[1/5] Initializing pipeline...")
    pipeline = WaterloggingPipeline(config)
    
    # Load models if checkpoints provided
    if detection_checkpoint:
        print(f"\n[2/5] Loading detection model from {detection_checkpoint}...")
        pipeline.load_detection_model(detection_checkpoint)
    else:
        print("\n[2/5] Using pretrained detection model (ImageNet weights)...")
    
    if forecast_checkpoint:
        print(f"\n[3/5] Loading forecasting model from {forecast_checkpoint}...")
        # Note: You need to specify input_size based on your model
        pipeline.load_forecasting_model(forecast_checkpoint, input_size=12)
    else:
        print("\n[3/5] Using default forecasting model (untrained)...")
    
    # Process video
    print("\n[4/5] Processing video/images...")
    output_video_path = config.OUTPUT_DIR / "output_video.mp4"
    
    results = pipeline.process_video(
        video_path=video_path,
        weather_data_path=weather_data_path,
        output_video_path=str(output_video_path),
        visualize=True,
        save_frames=True
    )
    
    print("\n[5/5] Processing complete!")
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Frames Processed: {results['frames_processed']}")
    print(f"Average FPS: {results['performance']['avg_fps']:.2f}")
    print(f"Total Time: {results['performance']['total_time']:.2f}s")
    print(f"\nRisk Statistics:")
    print(f"  Mean Fused Risk: {results['summary']['fused_risk']['mean']:.2%}")
    print(f"  Max Fused Risk: {results['summary']['fused_risk']['max']:.2%}")
    print(f"  Min Fused Risk: {results['summary']['fused_risk']['min']:.2%}")
    print(f"\nRisk Level Distribution:")
    print(f"  Low: {results['summary']['risk_levels']['low']} frames")
    print(f"  Medium: {results['summary']['risk_levels']['medium']} frames")
    print(f"  High: {results['summary']['risk_levels']['high']} frames")
    print("\n" + "="*60)
    print(f"\nOutput saved to: {config.OUTPUT_DIR}")
    print(f"  - Video: {output_video_path}")
    print(f"  - Frames: {config.OUTPUT_DIR}/frames/")
    print(f"  - Results: {config.OUTPUT_DIR}/*.json")
    print(f"  - Logs: {config.LOGS_DIR}/")
    print("="*60 + "\n")


def create_sample_data():
    """Create sample weather data for testing"""
    print("Creating sample weather data...")
    
    weather_loader = WeatherDataLoader()
    sample_path = config.DATA_DIR / "sample_weather.csv"
    
    weather_loader.create_sample_weather_data(
        output_path=str(sample_path),
        num_records=100
    )
    
    print(f"Sample weather data created at: {sample_path}")
    return str(sample_path)


def main():
    parser = argparse.ArgumentParser(
        description="AI-CCTV Waterlogging Detection & Forecasting Demo"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=False,
        help="Path to video file or image directory"
    )
    
    parser.add_argument(
        "--weather",
        type=str,
        default=None,
        help="Path to weather data CSV/JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--detection-model",
        type=str,
        default=None,
        help="Path to detection model checkpoint"
    )
    
    parser.add_argument(
        "--forecast-model",
        type=str,
        default=None,
        help="Path to forecasting model checkpoint"
    )
    
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample weather data for testing"
    )
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_data()
        return
    
    if not args.video:
        print("Error: --video argument is required (unless using --create-sample-data)")
        print("\nUsage examples:")
        print("  python demo.py --video path/to/video.mp4")
        print("  python demo.py --video path/to/images/ --weather weather.csv")
        print("  python demo.py --create-sample-data")
        return
    
    # Create sample weather data if not provided
    weather_path = args.weather
    if not weather_path:
        print("No weather data provided. Creating sample data...")
        weather_path = create_sample_data()
    
    # Run demo
    run_demo_video(
        video_path=args.video,
        weather_data_path=weather_path,
        output_dir=args.output,
        detection_checkpoint=args.detection_model,
        forecast_checkpoint=args.forecast_model
    )


if __name__ == "__main__":
    main()
