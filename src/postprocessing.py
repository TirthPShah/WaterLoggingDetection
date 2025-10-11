"""
Postprocessing module for segmentation outputs
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Dict, List


class MaskPostprocessor:
    """Postprocess segmentation masks"""
    
    def __init__(
        self,
        kernel_size: int = 5,
        min_area: int = 100,
        apply_morphology: bool = True,
        apply_filtering: bool = True
    ):
        """
        Initialize mask postprocessor
        
        Args:
            kernel_size: Size of morphological operation kernel
            min_area: Minimum area (pixels) to keep as waterlogging region
            apply_morphology: Apply morphological operations
            apply_filtering: Apply area-based filtering
        """
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.apply_morphology = apply_morphology
        self.apply_filtering = apply_filtering
        
        # Create morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
    
    def postprocess(
        self,
        mask: np.ndarray,
        probability_map: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Postprocess binary mask
        
        Args:
            mask: Binary mask (H, W) with values 0 or 1
            probability_map: Optional probability map for refinement
            
        Returns:
            Tuple of (cleaned_mask, statistics)
        """
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)
        
        # Apply morphological operations
        if self.apply_morphology:
            mask = self._apply_morphology(mask)
        
        # Filter small regions
        if self.apply_filtering:
            mask, num_regions_before, num_regions_after = self._filter_small_regions(mask)
        else:
            num_regions_before = num_regions_after = 0
        
        # Calculate statistics
        statistics = self._calculate_statistics(mask, probability_map)
        statistics['num_regions_before_filtering'] = num_regions_before
        statistics['num_regions_after_filtering'] = num_regions_after
        
        return mask, statistics
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Cleaned mask
        """
        # Close operation: fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Open operation: remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        return mask
    
    def _filter_small_regions(self, mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Remove small connected components
        
        Args:
            mask: Binary mask
            
        Returns:
            Tuple of (filtered_mask, num_regions_before, num_regions_after)
        """
        # Label connected components
        labeled_mask, num_features = ndimage.label(mask)
        num_regions_before = num_features
        
        # Calculate area of each component
        component_sizes = np.bincount(labeled_mask.ravel())
        
        # Create mask of components to keep
        keep_components = component_sizes >= self.min_area
        keep_components[0] = False  # Background
        
        # Filter mask
        filtered_mask = keep_components[labeled_mask].astype(np.uint8)
        
        # Count remaining regions
        _, num_regions_after = ndimage.label(filtered_mask)
        
        return filtered_mask, num_regions_before, num_regions_after
    
    def _calculate_statistics(
        self,
        mask: np.ndarray,
        probability_map: np.ndarray = None
    ) -> Dict:
        """
        Calculate mask statistics
        
        Args:
            mask: Binary mask
            probability_map: Optional probability map
            
        Returns:
            Dictionary of statistics
        """
        # Basic statistics
        total_pixels = mask.size
        waterlogged_pixels = np.sum(mask)
        waterlogged_ratio = waterlogged_pixels / total_pixels
        
        # Connected components
        labeled_mask, num_regions = ndimage.label(mask)
        
        statistics = {
            'total_pixels': int(total_pixels),
            'waterlogged_pixels': int(waterlogged_pixels),
            'waterlogged_ratio': float(waterlogged_ratio),
            'num_regions': int(num_regions),
        }
        
        # Region-specific statistics
        if num_regions > 0:
            region_sizes = []
            region_centroids = []
            
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_mask == region_id)
                region_size = np.sum(region_mask)
                region_sizes.append(region_size)
                
                # Calculate centroid
                coords = np.argwhere(region_mask)
                centroid = coords.mean(axis=0)
                region_centroids.append(centroid.tolist())
            
            statistics['region_sizes'] = region_sizes
            statistics['region_centroids'] = region_centroids
            statistics['largest_region_size'] = int(max(region_sizes))
            statistics['average_region_size'] = float(np.mean(region_sizes))
        else:
            statistics['region_sizes'] = []
            statistics['region_centroids'] = []
            statistics['largest_region_size'] = 0
            statistics['average_region_size'] = 0
        
        # Probability-based statistics
        if probability_map is not None:
            waterlogged_probs = probability_map[mask > 0]
            if len(waterlogged_probs) > 0:
                statistics['mean_confidence'] = float(waterlogged_probs.mean())
                statistics['min_confidence'] = float(waterlogged_probs.min())
                statistics['max_confidence'] = float(waterlogged_probs.max())
        
        return statistics
    
    def extract_region_features(
        self,
        mask: np.ndarray,
        image: np.ndarray = None
    ) -> List[Dict]:
        """
        Extract features for each waterlogged region
        
        Args:
            mask: Binary mask
            image: Optional original image for color/texture features
            
        Returns:
            List of feature dictionaries for each region
        """
        labeled_mask, num_regions = ndimage.label(mask)
        
        region_features = []
        
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_mask == region_id).astype(np.uint8)
            
            # Geometric features
            contours, _ = cv2.findContours(
                region_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            contour = contours[0]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Shape features
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
            
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            features = {
                'region_id': region_id,
                'area': float(area),
                'perimeter': float(perimeter),
                'circularity': float(circularity),
                'bounding_box': [int(x), int(y), int(w), int(h)],
                'aspect_ratio': float(aspect_ratio),
                'centroid': [int(cx), int(cy)]
            }
            
            # Color/texture features from original image
            if image is not None:
                region_pixels = image[region_mask > 0]
                if len(region_pixels) > 0:
                    features['mean_color'] = region_pixels.mean(axis=0).tolist()
                    features['std_color'] = region_pixels.std(axis=0).tolist()
            
            region_features.append(features)
        
        return region_features


class TemporalSmoother:
    """Apply temporal smoothing to reduce flickering in video sequences"""
    
    def __init__(self, alpha: float = 0.3, buffer_size: int = 5):
        """
        Initialize temporal smoother
        
        Args:
            alpha: Exponential smoothing factor (0-1)
            buffer_size: Number of frames to keep in buffer
        """
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.mask_buffer = []
        self.prob_buffer = []
    
    def smooth(
        self,
        current_mask: np.ndarray,
        current_prob: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal smoothing
        
        Args:
            current_mask: Current frame's mask
            current_prob: Current frame's probability map
            
        Returns:
            Tuple of (smoothed_mask, smoothed_prob)
        """
        # Add to buffer
        self.mask_buffer.append(current_mask)
        if current_prob is not None:
            self.prob_buffer.append(current_prob)
        
        # Maintain buffer size
        if len(self.mask_buffer) > self.buffer_size:
            self.mask_buffer.pop(0)
        if len(self.prob_buffer) > self.buffer_size:
            self.prob_buffer.pop(0)
        
        # Exponential smoothing
        if len(self.mask_buffer) == 1:
            smoothed_mask = current_mask
            smoothed_prob = current_prob if current_prob is not None else None
        else:
            # Weighted average with more weight on recent frames
            weights = np.array([self.alpha ** (len(self.mask_buffer) - i - 1) 
                               for i in range(len(self.mask_buffer))])
            weights /= weights.sum()
            
            # Smooth masks
            mask_stack = np.stack(self.mask_buffer, axis=0).astype(float)
            smoothed_mask = np.average(mask_stack, axis=0, weights=weights)
            smoothed_mask = (smoothed_mask > 0.5).astype(np.uint8)
            
            # Smooth probabilities
            if current_prob is not None and self.prob_buffer:
                prob_stack = np.stack(self.prob_buffer, axis=0)
                smoothed_prob = np.average(prob_stack, axis=0, weights=weights)
            else:
                smoothed_prob = current_prob
        
        return smoothed_mask, smoothed_prob
    
    def reset(self):
        """Reset buffer"""
        self.mask_buffer = []
        self.prob_buffer = []
