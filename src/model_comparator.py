"""
Model Comparison Report Generator

Compares Custom CNN vs Pretrained models across all metrics:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Inference Time
- Model Size
- Generates comprehensive comparison reports
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import numpy as np


class ModelComparator:
    """Compare multiple flood detection models."""
    
    def __init__(self):
        """Initialize comparator."""
        self.models_data = []
    
    def add_model(
        self,
        name: str,
        results_path: str,
        model_type: str,
        model_size_mb: float,
        training_time: float = None
    ):
        """
        Add a model's results for comparison.
        
        Args:
            name: Model display name
            results_path: Path to evaluation results JSON
            model_type: 'custom' or 'pretrained'
            model_size_mb: Model size in MB
            training_time: Training time in seconds
        """
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        model_data = {
            'name': name,
            'type': model_type,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'roc_auc': results['roc_auc'],
            'inference_time_ms': results['inference_time']['mean'] * 1000,
            'model_size_mb': model_size_mb,
            'training_time': training_time,
            'confusion_matrix': results['confusion_matrix']
        }
        
        self.models_data.append(model_data)
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table.
        
        Returns:
            DataFrame with all model comparisons
        """
        df = pd.DataFrame(self.models_data)
        
        # Reorder columns
        columns = [
            'name', 'type', 'accuracy', 'precision', 'recall', 
            'f1_score', 'roc_auc', 'inference_time_ms', 'model_size_mb'
        ]
        if 'training_time' in df.columns:
            columns.append('training_time')
        
        df = df[columns]
        
        # Format for display
        df_display = df.copy()
        for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}')
        df_display['inference_time_ms'] = df_display['inference_time_ms'].apply(lambda x: f'{x:.2f}')
        df_display['model_size_mb'] = df_display['model_size_mb'].apply(lambda x: f'{x:.2f}')
        
        return df, df_display
    
    def plot_metrics_comparison(self, save_path=None):
        """Plot comparison of all metrics across models."""
        df, _ = self.generate_comparison_table()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(self.models_data)
        
        for i, model_data in enumerate(self.models_data):
            values = [model_data[m] for m in metrics]
            offset = (i - len(self.models_data)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_data['name'], alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8
                )
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Metrics comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_efficiency_comparison(self, save_path=None):
        """Plot inference time vs model size."""
        df, _ = self.generate_comparison_table()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inference Time
        ax1.bar(df['name'], df['inference_time_ms'], color='steelblue', alpha=0.8)
        ax1.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Inference Speed Comparison', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(df['inference_time_ms']):
            ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Model Size
        ax2.bar(df['name'], df['model_size_mb'], color='coral', alpha=0.8)
        ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax2.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(df['model_size_mb']):
            ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Efficiency comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_accuracy_vs_efficiency(self, save_path=None):
        """Scatter plot: Accuracy vs Inference Time."""
        df, _ = self.generate_comparison_table()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'custom': 'blue', 'pretrained': 'red'}
        
        for model_type in df['type'].unique():
            mask = df['type'] == model_type
            ax.scatter(
                df[mask]['inference_time_ms'],
                df[mask]['accuracy'] * 100,
                s=df[mask]['model_size_mb'] * 10,
                c=colors.get(model_type, 'gray'),
                alpha=0.6,
                label=model_type.capitalize(),
                edgecolors='black',
                linewidths=1
            )
            
            # Add labels
            for _, row in df[mask].iterrows():
                ax.annotate(
                    row['name'],
                    (row['inference_time_ms'], row['accuracy'] * 100),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold'
                )
        
        ax.set_xlabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Inference Speed\n(Bubble size = Model Size)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Accuracy vs efficiency saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path='reports/model_comparison_report.md'):
        """Generate comprehensive markdown report."""
        df, df_display = self.generate_comparison_table()
        
        # Find best models for each metric
        best_accuracy = df.loc[df['accuracy'].idxmax()]['name']
        best_f1 = df.loc[df['f1_score'].idxmax()]['name']
        best_speed = df.loc[df['inference_time_ms'].idxmin()]['name']
        smallest = df.loc[df['model_size_mb'].idxmin()]['name']
        
        report = f"""# Flood Detection Model Comparison Report

## Executive Summary

This report compares the performance of multiple flood detection models:
- **Custom CNN** (trained from scratch)
- **Pretrained Models** (ResNet50, EfficientNet-B0, MobileNetV2)

### Key Findings

- **Best Accuracy**: {best_accuracy} ({df['accuracy'].max():.4f})
- **Best F1-Score**: {best_f1} ({df['f1_score'].max():.4f})
- **Fastest Inference**: {best_speed} ({df['inference_time_ms'].min():.2f} ms)
- **Smallest Model**: {smallest} ({df['model_size_mb'].min():.2f} MB)

---

## Performance Comparison Table

| Model | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference (ms) | Size (MB) |
|-------|------|----------|-----------|--------|----------|---------|----------------|-----------|
"""
        
        for _, row in df_display.iterrows():
            report += f"| {row['name']} | {row['type']} | {row['accuracy']} | {row['precision']} | {row['recall']} | {row['f1_score']} | {row['roc_auc']} | {row['inference_time_ms']} | {row['model_size_mb']} |\n"
        
        report += """
---

## Detailed Analysis

### 1. Accuracy Metrics

"""
        
        for model_data in self.models_data:
            report += f"""
#### {model_data['name']}
- **Accuracy**: {model_data['accuracy']:.4f} ({model_data['accuracy']*100:.2f}%)
- **Precision**: {model_data['precision']:.4f}
- **Recall**: {model_data['recall']:.4f}
- **F1-Score**: {model_data['f1_score']:.4f}
- **ROC-AUC**: {model_data['roc_auc']:.4f}
"""
        
        report += """
### 2. Efficiency Metrics

"""
        
        for model_data in self.models_data:
            report += f"""
#### {model_data['name']}
- **Inference Time**: {model_data['inference_time_ms']:.2f} ms/image
- **Model Size**: {model_data['model_size_mb']:.2f} MB
- **FPS**: {1000/model_data['inference_time_ms']:.2f}
"""
        
        report += """
---

## Recommendations

### For Production Deployment
"""
        
        # Recommendation logic
        if df['accuracy'].max() > 0.95:
            best_prod = df.loc[df['f1_score'].idxmax()]['name']
            report += f"\n**Recommended**: {best_prod}\n"
            report += f"- High accuracy ({df.loc[df['f1_score'].idxmax()]['accuracy']:.4f})\n"
            report += f"- Best F1-score ({df['f1_score'].max():.4f})\n"
        
        report += """
### For Edge Devices
"""
        
        # Find best trade-off for edge
        df['efficiency_score'] = (df['accuracy'] * 0.6) + (1 / df['model_size_mb'] * 0.2) + (1 / df['inference_time_ms'] * 0.2)
        best_edge = df.loc[df['efficiency_score'].idxmax()]['name']
        report += f"\n**Recommended**: {best_edge}\n"
        report += f"- Good accuracy with small size and fast inference\n"
        
        report += """
---

## Conclusion

This comprehensive comparison demonstrates the trade-offs between model architectures.
Custom CNNs offer good performance with smaller size, while pretrained models leverage
transfer learning for potentially better accuracy at the cost of larger model size.

"""
        
        # Save report
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Comparison report saved to: {save_path}")
        
        return report
    
    def print_summary(self):
        """Print comparison summary to console."""
        df, df_display = self.generate_comparison_table()
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        
        print("\n📊 Performance Metrics:")
        print(df_display[['name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))
        
        print("\n⚡ Efficiency Metrics:")
        print(df_display[['name', 'inference_time_ms', 'model_size_mb']].to_string(index=False))
        
        print("\n🏆 Winners:")
        print(f"  Best Accuracy:  {df.loc[df['accuracy'].idxmax()]['name']} ({df['accuracy'].max():.4f})")
        print(f"  Best F1-Score:  {df.loc[df['f1_score'].idxmax()]['name']} ({df['f1_score'].max():.4f})")
        print(f"  Fastest:        {df.loc[df['inference_time_ms'].idxmin()]['name']} ({df['inference_time_ms'].min():.2f} ms)")
        print(f"  Smallest:       {df.loc[df['model_size_mb'].idxmin()]['name']} ({df['model_size_mb'].min():.2f} MB)")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Model Comparator - Ready for use")
    print("\nUsage example:")
    print("""
    comparator = ModelComparator()
    comparator.add_model('Custom CNN', 'results/custom_results.json', 'custom', 6.62)
    comparator.add_model('ResNet50', 'results/resnet50_results.json', 'pretrained', 91.8)
    comparator.print_summary()
    comparator.generate_report()
    """)
