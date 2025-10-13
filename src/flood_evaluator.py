"""
Evaluation framework for flood detection models.

Includes:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve and AUC
- Inference time benchmarking
- Model size analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from pathlib import Path
import time
import json
from typing import Dict, Tuple, List
from tqdm import tqdm


class FloodDetectionEvaluator:
    """Comprehensive evaluation for flood detection models."""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        self.results = {}
    
    def evaluate(
        self,
        dataloader,
        class_names=['Non-Flooded', 'Flooded']
    ) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            dataloader: Test data loader
            class_names: Class names for display
            
        Returns:
            Dictionary with all metrics
        """
        print("=" * 70)
        print("Running Comprehensive Evaluation")
        print("=" * 70)
        
        # Collect predictions and labels
        all_preds = []
        all_labels = []
        all_probs = []
        inference_times = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Evaluating'):
                images = images.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / images.size(0))  # Per image
                
                # Collect results
                probs = outputs.cpu().numpy().squeeze()
                preds = (outputs > 0.5).float().cpu().numpy().squeeze()
                
                all_probs.extend(probs if probs.ndim > 0 else [probs])
                all_preds.extend(preds if preds.ndim > 0 else [preds])
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        results = {}
        
        # 1. Basic metrics
        results['accuracy'] = accuracy_score(all_labels, all_preds)
        results['precision'] = precision_score(all_labels, all_preds, zero_division=0)
        results['recall'] = recall_score(all_labels, all_preds, zero_division=0)
        results['f1_score'] = f1_score(all_labels, all_preds, zero_division=0)
        
        # 2. Confusion matrix
        results['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
        
        # 3. ROC and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        results['roc_auc'] = auc(fpr, tpr)
        results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        
        # 4. Per-class metrics
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        results['per_class'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        # 5. Inference time
        results['inference_time'] = {
            'mean': float(np.mean(inference_times)),
            'std': float(np.std(inference_times)),
            'min': float(np.min(inference_times)),
            'max': float(np.max(inference_times))
        }
        
        # 6. Classification report
        results['classification_report'] = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True
        )
        
        self.results = results
        
        # Print summary
        self._print_summary(class_names)
        
        return results
    
    def _print_summary(self, class_names):
        """Print evaluation summary."""
        print("\n" + "-" * 70)
        print("Evaluation Results")
        print("-" * 70)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy:  {self.results['accuracy']:.4f} ({self.results['accuracy']*100:.2f}%)")
        print(f"  Precision: {self.results['precision']:.4f}")
        print(f"  Recall:    {self.results['recall']:.4f}")
        print(f"  F1-Score:  {self.results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {self.results['roc_auc']:.4f}")
        
        print(f"\n🎯 Per-Class Metrics:")
        pc = self.results['per_class']
        print(f"  Sensitivity (TPR): {pc['sensitivity']:.4f}")
        print(f"  Specificity (TNR): {pc['specificity']:.4f}")
        
        print(f"\n⏱️  Inference Time (per image):")
        it = self.results['inference_time']
        print(f"  Mean: {it['mean']*1000:.2f} ms")
        print(f"  Std:  {it['std']*1000:.2f} ms")
        
        print(f"\n📈 Confusion Matrix:")
        cm = self.results['confusion_matrix']
        print(f"                Predicted")
        print(f"              {class_names[0]:^12} {class_names[1]:^12}")
        print(f"  Actual")
        print(f"  {class_names[0]:^12} {cm[0,0]:^12} {cm[0,1]:^12}")
        print(f"  {class_names[1]:^12} {cm[1,0]:^12} {cm[1,1]:^12}")
    
    def plot_confusion_matrix(self, save_path=None, class_names=['Non-Flooded', 'Flooded']):
        """Plot confusion matrix."""
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve."""
        roc = self.results['roc_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            roc['fpr'], roc['tpr'],
            color='darkorange', lw=2,
            label=f"ROC Curve (AUC = {self.results['roc_auc']:.4f})"
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ ROC curve saved to: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None):
        """Plot bar chart of key metrics."""
        metrics = {
            'Accuracy': self.results['accuracy'],
            'Precision': self.results['precision'],
            'Recall': self.results['recall'],
            'F1-Score': self.results['f1_score'],
            'ROC-AUC': self.results['roc_auc']
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values(), color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        plt.ylim([0, 1.1])
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.xticks(rotation=15, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Metrics comparison saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, save_path):
        """Save evaluation results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in self.results.items():
            if key == 'confusion_matrix':
                results_json[key] = value.tolist()
            elif key == 'roc_curve':
                results_json[key] = {
                    'fpr': value['fpr'].tolist(),
                    'tpr': value['tpr'].tolist(),
                    'thresholds': value['thresholds'].tolist()
                }
            else:
                results_json[key] = value
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"✅ Results saved to: {save_path}")
    
    def benchmark_inference(self, input_size=(1, 3, 244, 244), num_iterations=100):
        """
        Benchmark inference speed.
        
        Args:
            input_size: Input tensor size
            num_iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with timing statistics
        """
        print("\n" + "=" * 70)
        print("Benchmarking Inference Speed")
        print("=" * 70)
        
        # Warmup
        dummy_input = torch.randn(input_size).to(self.device)
        for _ in range(10):
            _ = self.model(dummy_input)
        
        # Actual benchmark
        times = []
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc='Benchmarking'):
                start = time.time()
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        times = np.array(times) * 1000  # Convert to milliseconds
        
        results = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'fps': 1000.0 / np.mean(times)
        }
        
        print(f"\n📊 Inference Speed:")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  Std:  {results['std_ms']:.2f} ms")
        print(f"  FPS:  {results['fps']:.2f}")
        
        return results


def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'size_mb': size_mb,
        'param_size_mb': param_size / (1024 ** 2),
        'buffer_size_mb': buffer_size / (1024 ** 2)
    }


if __name__ == "__main__":
    # Test evaluator
    from src.flood_classifier import create_model
    
    print("Testing Flood Detection Evaluator")
    
    # Create dummy model and data
    model = create_model('custom')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Benchmark
    evaluator = FloodDetectionEvaluator(model, device)
    benchmark_results = evaluator.benchmark_inference()
    
    # Model size
    size_info = get_model_size(model)
    print(f"\n📦 Model Size: {size_info['size_mb']:.2f} MB")
    
    print("\n✅ Evaluator test completed!")
