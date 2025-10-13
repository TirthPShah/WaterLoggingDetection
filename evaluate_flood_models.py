"""
Comprehensive evaluation script for flood detection models.

Evaluates all trained models and generates:
- Performance metrics (Accuracy, F1, ROC-AUC, etc.)
- Confusion matrices
- ROC curves
- Grad-CAM visualizations
- Model comparison report
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import json

from src.flood_classifier import create_model
from src.flood_preprocessing import get_flood_detection_transforms
from src.flood_evaluator import FloodDetectionEvaluator, get_model_size
from src.gradcam_visualizer import FloodGradCAMVisualizer
from src.model_comparator import ModelComparator


def evaluate_single_model(
    model_path: str,
    model_type: str,
    model_name: str,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str
):
    """
    Evaluate a single model comprehensively.
    
    Args:
        model_path: Path to model checkpoint
        model_type: 'custom' or 'pretrained'
        model_name: Model name
        test_loader: Test data loader
        device: Device to evaluate on
        output_dir: Output directory
        
    Returns:
        Dictionary with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"Evaluating: {model_name}")
    print("=" * 70)
    
    # Load model
    if model_type == 'custom':
        model = create_model('custom')
    else:
        # Extract base model name from full name
        base_name = model_name.lower().replace(' ', '_')
        for mn in ['resnet50', 'efficientnet_b0', 'mobilenet_v2']:
            if mn in base_name:
                model = create_model('pretrained', model_name=mn, pretrained=False)
                break
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"✅ Model loaded from: {model_path}")
    
    # Get model size
    size_info = get_model_size(model)
    print(f"📦 Model size: {size_info['size_mb']:.2f} MB")
    
    # 1. Comprehensive Evaluation
    evaluator = FloodDetectionEvaluator(model, device)
    results = evaluator.evaluate(test_loader)
    
    # Save metrics
    evaluator.save_results(output_dir / 'evaluation_results.json')
    
    # 2. Plot visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrix(save_path=output_dir / 'confusion_matrix.png')
    evaluator.plot_roc_curve(save_path=output_dir / 'roc_curve.png')
    evaluator.plot_metrics_comparison(save_path=output_dir / 'metrics.png')
    
    # 3. Benchmark inference
    print("\nBenchmarking inference speed...")
    benchmark = evaluator.benchmark_inference(num_iterations=100)
    
    # 4. Grad-CAM Visualization (if possible)
    try:
        print("\nGenerating Grad-CAM visualizations...")
        gradcam_viz = FloodGradCAMVisualizer(model, device)
        gradcam_viz.visualize_batch(
            test_loader,
            num_samples=5,
            save_dir=output_dir / 'gradcam'
        )
    except Exception as e:
        print(f"⚠️  Grad-CAM generation failed: {e}")
    
    # Compile results
    full_results = {
        'model_name': model_name,
        'model_type': model_type,
        'metrics': results,
        'benchmark': benchmark,
        'model_size_mb': size_info['size_mb']
    }
    
    # Save complete results
    with open(output_dir / 'complete_results.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n✅ Evaluation complete for {model_name}")
    print(f"   Results saved to: {output_dir}")
    
    return full_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Flood Detection Models')
    
    # Data arguments
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Path to test dataset (ImageFolder format)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='output/evaluations',
                       help='Output directory for results')
    
    # Evaluation options
    parser.add_argument('--compare', action='store_true',
                       help='Generate model comparison report')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 70)
    print("FLOOD DETECTION MODEL EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Test data: {args.test_dir}")
    
    # Load test data
    print("\nLoading test dataset...")
    test_transform = get_flood_detection_transforms(train=False)
    test_dataset = ImageFolder(root=args.test_dir, transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✅ Test dataset loaded")
    print(f"   Total samples: {len(test_dataset)}")
    print(f"   Classes: {test_dataset.classes}")
    
    # Find all trained models
    models_dir = Path(args.models_dir)
    
    model_configs = []
    
    # Custom CNN
    custom_path = models_dir / 'custom_cnn' / 'best_model.pth'
    if custom_path.exists():
        model_configs.append({
            'path': str(custom_path),
            'type': 'custom',
            'name': 'Custom CNN',
            'output_dir': Path(args.output_dir) / 'custom_cnn'
        })
    
    # Pretrained models
    pretrained_configs = [
        ('resnet50', 'ResNet50'),
        ('efficientnet_b0', 'EfficientNet-B0'),
        ('mobilenet_v2', 'MobileNetV2')
    ]
    
    for model_dir, display_name in pretrained_configs:
        model_path = models_dir / model_dir / 'best_model.pth'
        if model_path.exists():
            model_configs.append({
                'path': str(model_path),
                'type': 'pretrained',
                'name': display_name,
                'output_dir': Path(args.output_dir) / model_dir
            })
    
    if not model_configs:
        print("\n❌ No trained models found!")
        print(f"   Expected models in: {models_dir}")
        print("\nTrain models first using train_flood_classifier.py")
        return
    
    print(f"\n✅ Found {len(model_configs)} trained models:")
    for config in model_configs:
        print(f"   - {config['name']}")
    
    # Evaluate each model
    all_results = []
    
    for config in model_configs:
        results = evaluate_single_model(
            model_path=config['path'],
            model_type=config['type'],
            model_name=config['name'],
            test_loader=test_loader,
            device=device,
            output_dir=str(config['output_dir'])
        )
        all_results.append(results)
    
    # Generate comparison if requested
    if args.compare and len(all_results) > 1:
        print("\n" + "=" * 70)
        print("GENERATING MODEL COMPARISON REPORT")
        print("=" * 70)
        
        comparator = ModelComparator()
        
        for result in all_results:
            output_dir = Path(args.output_dir) / result['model_name'].lower().replace(' ', '_').replace('-', '_')
            comparator.add_model(
                name=result['model_name'],
                results_path=str(output_dir / 'evaluation_results.json'),
                model_type=result['model_type'],
                model_size_mb=result['model_size_mb']
            )
        
        # Print summary
        comparator.print_summary()
        
        # Generate visualizations
        comp_dir = Path(args.output_dir) / 'comparison'
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        comparator.plot_metrics_comparison(save_path=comp_dir / 'metrics_comparison.png')
        comparator.plot_efficiency_comparison(save_path=comp_dir / 'efficiency_comparison.png')
        comparator.plot_accuracy_vs_efficiency(save_path=comp_dir / 'accuracy_vs_efficiency.png')
        
        # Generate report
        comparator.generate_report(save_path=comp_dir / 'comparison_report.md')
        
        print(f"\n✅ Comparison report generated in: {comp_dir}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review evaluation metrics in output/evaluations/*/")
    print("  2. Check Grad-CAM visualizations for model interpretability")
    print("  3. Read comparison report: output/evaluations/comparison/comparison_report.md")
    print("  4. Select best model based on your requirements")


if __name__ == "__main__":
    main()
