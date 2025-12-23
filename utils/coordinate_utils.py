# Coordinate Utilities
"""
Coordinate conversion and bounding box utilities for OCR Pipeline.
"""

from typing import List, Tuple

from config import BBOX_NORMALIZATION_RANGE


class CoordinateConverter:
    """Handle coordinate normalization and containment checks"""
    
    @staticmethod
    def normalize_bbox(bbox_pixels: List[int], img_width: int, img_height: int) -> List[int]:
        """Convert pixel coordinates to normalized 0-1024 range"""
        x0, y0, x1, y1 = bbox_pixels
        return [
            int((x0 / img_width) * BBOX_NORMALIZATION_RANGE),
            int((y0 / img_height) * BBOX_NORMALIZATION_RANGE),
            int((x1 / img_width) * BBOX_NORMALIZATION_RANGE),
            int((y1 / img_height) * BBOX_NORMALIZATION_RANGE)
        ]
    
    @staticmethod
    def validate_bbox(bbox: List[int], min_size: int = 5) -> bool:
        """Validate that bounding box has minimum size"""
        x0, y0, x1, y1 = bbox
        return (x1 - x0) >= min_size and (y1 - y0) >= min_size
    
    @staticmethod
    def is_contained(child_bbox: List[int], parent_bbox: List[int]) -> bool:
        """
        Check if child box is COMPLETELY contained within parent box.
        Returns True only if 100% of the child is inside the parent.
        """
        cx0, cy0, cx1, cy1 = child_bbox
        px0, py0, px1, py1 = parent_bbox
        return (cx0 >= px0 and cy0 >= py0 and cx1 <= px1 and cy1 <= py1)
    
    @staticmethod
    def overlaps_with(bbox1: List[int], bbox2: List[int]) -> bool:
        """Check if two bounding boxes overlap (any intersection)"""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)
    
    @staticmethod
    def bbox_area(bbox: List[int]) -> int:
        """Calculate area of bounding box"""
        x0, y0, x1, y1 = bbox
        return max(0, x1 - x0) * max(0, y1 - y0)
    
    @staticmethod
    def intersection_area(bbox1: List[int], bbox2: List[int]) -> int:
        """Calculate intersection area between two bounding boxes"""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        
        inter_x0 = max(x0_1, x0_2)
        inter_y0 = max(y0_1, y0_2)
        inter_x1 = min(x1_1, x1_2)
        inter_y1 = min(y1_1, y1_2)
        
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0
        return (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    
    @staticmethod
    def iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IOU) between two bounding boxes"""
        area1 = CoordinateConverter.bbox_area(bbox1)
        area2 = CoordinateConverter.bbox_area(bbox2)
        inter = CoordinateConverter.intersection_area(bbox1, bbox2)
        
        if area1 + area2 - inter == 0:
            return 0.0
        return inter / (area1 + area2 - inter)
    
    @staticmethod
    def merge_bboxes(bbox1: List[int], bbox2: List[int]) -> List[int]:
        """Merge two bounding boxes into their union (enclosing box)"""
        return [
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3])
        ]
    
    @staticmethod
    def crop_to_full_coords(crop_bbox: List[int], parent_offset: Tuple[int, int]) -> List[int]:
        """Convert crop-relative coordinates to full image coordinates"""
        x0, y0, x1, y1 = crop_bbox
        ox, oy = parent_offset
        return [x0 + ox, y0 + oy, x1 + ox, y1 + oy]
