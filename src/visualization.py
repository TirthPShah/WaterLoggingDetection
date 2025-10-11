"""
Visualization module for waterlogging detection and risk maps
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional, Dict
from pathlib import Path


class WaterloggingVisualizer:
    """Visualize waterlogging detection and forecasting results"""
    
    def __init__(
        self,
        overlay_alpha: float = 0.4,
        colormap: str = "jet"
    ):
        """
        Initialize visualizer
        
        Args:
            overlay_alpha: Transparency of overlay (0-1)
            colormap: Matplotlib colormap name
        """
        self.overlay_alpha = overlay_alpha
        self.colormap = colormap
        
        # Color schemes
        self.risk_colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 165, 255),  # Orange
            'high': (0, 0, 255)       # Red
        }
    
    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Overlay binary mask on image
        
        Args:
            image: Original image (H, W, C) in RGB or BGR
            mask: Binary mask (H, W)
            color: Overlay color in BGR format
            alpha: Transparency (uses default if None)
            
        Returns:
            Overlayed image
        """
        alpha = alpha if alpha is not None else self.overlay_alpha
        
        # Ensure image is in correct format
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Create colored overlay
        overlay = image.copy()
        overlay[mask > 0] = color
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def overlay_probability_map(
        self,
        image: np.ndarray,
        probability_map: np.ndarray,
        alpha: Optional[float] = None,
        colormap: Optional[str] = None
    ) -> np.ndarray:
        """
        Overlay probability/risk map with heatmap visualization
        
        Args:
            image: Original image (H, W, C)
            probability_map: Probability values (H, W) in range [0, 1]
            alpha: Transparency
            colormap: Colormap name
            
        Returns:
            Overlayed image with heatmap
        """
        alpha = alpha if alpha is not None else self.overlay_alpha
        colormap = colormap if colormap is not None else self.colormap
        
        # Resize probability map if needed
        if image.shape[:2] != probability_map.shape:
            probability_map = cv2.resize(
                probability_map,
                (image.shape[1], image.shape[0])
            )
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored_map = cmap(probability_map)[:, :, :3]  # Remove alpha channel
        colored_map = (colored_map * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR)
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - alpha, colored_map, alpha, 0)
        
        return result
    
    def create_risk_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        risk_level: str,
        fused_risk: float,
        detection_risk: float,
        forecast_risk: float,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Create comprehensive risk visualization with annotations
        
        Args:
            image: Original image
            mask: Detection mask
            risk_level: Risk level (low, medium, high)
            fused_risk: Fused risk score
            detection_risk: Detection risk score
            forecast_risk: Forecast risk score
            metadata: Additional metadata to display
            
        Returns:
            Annotated visualization image
        """
        # Overlay mask
        color = self.risk_colors.get(risk_level, (0, 0, 255))
        vis_image = self.overlay_mask(image, mask, color=color)
        
        # Add risk information panel
        vis_image = self._add_info_panel(
            vis_image,
            risk_level,
            fused_risk,
            detection_risk,
            forecast_risk,
            metadata
        )
        
        # Draw contours around waterlogged regions
        vis_image = self._draw_region_contours(vis_image, mask)
        
        return vis_image
    
    def _add_info_panel(
        self,
        image: np.ndarray,
        risk_level: str,
        fused_risk: float,
        detection_risk: float,
        forecast_risk: float,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """Add information panel to image"""
        h, w = image.shape[:2]
        panel_height = 150
        panel_width = w
        
        # Create panel
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark gray background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Risk level
        risk_color = self.risk_colors.get(risk_level, (255, 255, 255))
        cv2.putText(
            panel,
            f"Risk Level: {risk_level.upper()}",
            (10, 30),
            font,
            font_scale,
            risk_color,
            thickness
        )
        
        # Scores
        cv2.putText(
            panel,
            f"Fused Risk: {fused_risk:.2%}",
            (10, 60),
            font,
            font_scale - 0.1,
            (255, 255, 255),
            thickness - 1
        )
        
        cv2.putText(
            panel,
            f"Detection: {detection_risk:.2%}",
            (10, 85),
            font,
            font_scale - 0.1,
            (200, 200, 200),
            thickness - 1
        )
        
        cv2.putText(
            panel,
            f"Forecast: {forecast_risk:.2%}",
            (10, 110),
            font,
            font_scale - 0.1,
            (200, 200, 200),
            thickness - 1
        )
        
        # Additional metadata
        if metadata:
            num_regions = metadata.get('detection_metadata', {}).get('num_regions', 0)
            cv2.putText(
                panel,
                f"Regions: {num_regions}",
                (panel_width - 200, 30),
                font,
                font_scale - 0.1,
                (255, 255, 255),
                thickness - 1
            )
            
            timestamp = metadata.get('timestamp', '')
            if timestamp:
                time_str = timestamp.split('T')[1][:8] if 'T' in timestamp else ''
                cv2.putText(
                    panel,
                    f"Time: {time_str}",
                    (panel_width - 200, 60),
                    font,
                    font_scale - 0.1,
                    (200, 200, 200),
                    thickness - 1
                )
        
        # Concatenate panel with image
        result = np.vstack([panel, image])
        
        return result
    
    def _draw_region_contours(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw contours around detected regions"""
        # Resize mask if needed
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw contours
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        
        return result
    
    def create_comparison_view(
        self,
        images: list,
        titles: list,
        rows: int = 1,
        cols: int = None
    ) -> np.ndarray:
        """
        Create side-by-side comparison of multiple images
        
        Args:
            images: List of images to compare
            titles: List of titles for each image
            rows: Number of rows in grid
            cols: Number of columns (auto-calculated if None)
            
        Returns:
            Combined comparison image
        """
        n_images = len(images)
        
        if cols is None:
            cols = (n_images + rows - 1) // rows
        
        # Resize all images to same size
        target_h, target_w = images[0].shape[:2]
        resized_images = []
        
        for img in images:
            if img.shape[:2] != (target_h, target_w):
                img_resized = cv2.resize(img, (target_w, target_h))
            else:
                img_resized = img
            resized_images.append(img_resized)
        
        # Add titles
        titled_images = []
        for img, title in zip(resized_images, titles):
            img_with_title = self._add_title(img, title)
            titled_images.append(img_with_title)
        
        # Arrange in grid
        grid_rows = []
        for i in range(rows):
            row_images = titled_images[i*cols:(i+1)*cols]
            if row_images:
                # Pad with black images if needed
                while len(row_images) < cols:
                    black_img = np.zeros_like(titled_images[0])
                    row_images.append(black_img)
                row = np.hstack(row_images)
                grid_rows.append(row)
        
        result = np.vstack(grid_rows)
        
        return result
    
    def _add_title(
        self,
        image: np.ndarray,
        title: str,
        bg_color: Tuple[int, int, int] = (40, 40, 40)
    ) -> np.ndarray:
        """Add title bar to image"""
        title_height = 40
        h, w = image.shape[:2]
        
        # Create title bar
        title_bar = np.zeros((title_height, w, 3), dtype=np.uint8)
        title_bar[:] = bg_color
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Center text
        (text_width, text_height), _ = cv2.getTextSize(title, font, font_scale, thickness)
        x = (w - text_width) // 2
        y = (title_height + text_height) // 2
        
        cv2.putText(
            title_bar,
            title,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        # Concatenate
        result = np.vstack([title_bar, image])
        
        return result
    
    def create_temporal_plot(
        self,
        prediction_history: list,
        output_path: str
    ):
        """
        Create temporal plot of risk predictions
        
        Args:
            prediction_history: List of prediction dictionaries
            output_path: Path to save plot
        """
        if not prediction_history:
            print("No prediction history available")
            return
        
        # Extract data
        timestamps = range(len(prediction_history))
        fused_risks = [p['fused_risk'] for p in prediction_history]
        detection_risks = [p['detection_risk'] for p in prediction_history]
        forecast_risks = [p['forecast_risk'] for p in prediction_history]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(timestamps, fused_risks, 'b-', linewidth=2, label='Fused Risk')
        plt.plot(timestamps, detection_risks, 'g--', linewidth=1.5, label='Detection Risk')
        plt.plot(timestamps, forecast_risks, 'r--', linewidth=1.5, label='Forecast Risk')
        
        # Add risk level zones
        plt.axhspan(0.0, 0.3, alpha=0.1, color='green', label='Low Risk')
        plt.axhspan(0.3, 0.6, alpha=0.1, color='orange', label='Medium Risk')
        plt.axhspan(0.6, 1.0, alpha=0.1, color='red', label='High Risk')
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Risk Score', fontsize=12)
        plt.title('Waterlogging Risk Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal plot saved to {output_path}")
    
    def save_visualization(
        self,
        image: np.ndarray,
        output_path: str
    ):
        """Save visualization to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR if needed (OpenCV expects BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume input is RGB, convert to BGR for saving
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        cv2.imwrite(str(output_path), image_bgr)
    
    def create_video_from_frames(
        self,
        frames: list,
        output_path: str,
        fps: int = 30
    ):
        """
        Create video from list of frames
        
        Args:
            frames: List of frame images
            output_path: Output video path
            fps: Frames per second
        """
        if not frames:
            print("No frames to create video")
            return
        
        h, w = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frames:
            # Ensure frame size matches
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            
            # Convert RGB to BGR if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to {output_path}")
