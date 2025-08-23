import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

class GearInspector:
    def __init__(self, image_path):
        self.ideal_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.ideal_image is None:   
            print("Image cannot be loaded from path:", image_path)
            return
        self.ideal_blur = cv2.GaussianBlur(self.ideal_image, (5, 5), 0)
        _, self.ideal_thresh = cv2.threshold(self.ideal_blur, 127, 255, cv2.THRESH_BINARY)
        self.mask = self.create_gear_mask(self.ideal_thresh)
        self.ideal_masked = cv2.bitwise_and(self.ideal_image, self.ideal_image, mask=self.mask)
        self.ideal_inner_diameter = self.measure_inner_diameter(self.ideal_thresh)
    
    def create_gear_mask(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found in the image.")
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        mask = np.zeros_like(image)
        cv2.circle(mask, center, radius, 255, -1)
        return mask

    def measure_inner_diameter(self, thresh_image):
        contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:  # Need at least outer gear and inner hole
            return 0
        

        contours_with_area = [(cv2.contourArea(c), c) for c in contours]
        contours_with_area.sort(key=lambda x: x[0], reverse=True)
        
        # The largest should be the outer gear, second largest should be inner hole
        if len(contours_with_area) >= 2:
            inner_area = contours_with_area[1][0]
            
            if inner_area > 1000:  # Minimum area threshold
                diameter = 2 * np.sqrt(inner_area / np.pi)
                return diameter
        
        # Fallback method: look for holes by inverting image
        inverted = cv2.bitwise_not(thresh_image)
        hole_contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if hole_contours:
            # Find largest hole
            largest_hole = max(hole_contours, key=cv2.contourArea)
            hole_area = cv2.contourArea(largest_hole)
            
            if hole_area > 500:  # Reasonable minimum
                diameter = 2 * np.sqrt(hole_area / np.pi)
                return diameter
        
        return 0

    def inspect_gear(self, test_image_path):
        sample_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE) 
        if sample_image is None:
            print("Image cannot be loaded from path:", test_image_path)
            return None, None, None, None
        
        sample_blur = cv2.GaussianBlur(sample_image, (5, 5), 0)
        _, sample_thresh = cv2.threshold(sample_blur, 127, 255, cv2.THRESH_BINARY)
        sample_mask = self.create_gear_mask(sample_thresh)
        
        if sample_mask is None:
            return None, None, None, None
        
        sample_masked = cv2.bitwise_and(sample_image, sample_image, mask=sample_mask)
        
        # Tooth damage detection (your original logic)
        difference = cv2.absdiff(self.ideal_masked, sample_masked)
        _, diff_thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        broken_teeth = 0
        worn_teeth = 0
        min_tooth_area = 100 
        broken_threshold = 500

        # Simple approach: just count significant differences
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_tooth_area:
                if area > broken_threshold:
                    broken_teeth += 1
                else:
                    worn_teeth += 1
        
        # BONUS: Inner diameter comparison - completely new approach
        sample_inner_diameter = self.measure_inner_diameter(sample_thresh)
        inner_diameter_status = "identical"
        
        print(f"Ideal inner diameter: {self.ideal_inner_diameter:.2f}, Sample inner diameter: {sample_inner_diameter:.2f}")
        
        diameter_diff = abs(sample_inner_diameter - self.ideal_inner_diameter)
        tolerance = 2.0  # Allow 2 pixel difference
        
        if diameter_diff > tolerance:
            if sample_inner_diameter > self.ideal_inner_diameter:
                inner_diameter_status = "larger"
            else:
                inner_diameter_status = "smaller"
        
        # Prepare result visualization
        result_vis = self.prepare_visualization(sample_image, diff_thresh, broken_teeth, worn_teeth, inner_diameter_status)
        
        return broken_teeth, worn_teeth, inner_diameter_status, result_vis

    def prepare_visualization(self, sample, diff, broken, worn, inner_status):
        # Convert images to color for visualization
        ideal_color = cv2.cvtColor(self.ideal_image, cv2.COLOR_GRAY2BGR)
        sample_color = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        
        # Highlight differences in red
        diff_color[diff > 0] = [0, 0, 255]
        
        # Add text to the difference image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(diff_color, f"Broken: {broken}, Worn: {worn}", (10, 30), 
                   font, 0.7, (0, 255, 0), 2)
        cv2.putText(diff_color, f"Inner: {inner_status}", (10, 60), 
                   font, 0.7, (0, 255, 0), 2)
        
        # Combine images for side-by-side comparison
        top_row = np.hstack((ideal_color, sample_color))
        bottom_row = np.hstack((diff_color, np.zeros_like(diff_color)))
        
        # If the images have different heights, resize them
        if top_row.shape[0] != bottom_row.shape[0]:
            bottom_row = cv2.resize(bottom_row, (top_row.shape[1], top_row.shape[0]))
        
        result_vis = np.vstack((top_row, bottom_row))
        
        return result_vis


inspector = GearInspector('./samples/ideal.jpg')
samples = ['./samples/sample2.jpg', './samples/sample3.jpg', './samples/sample4.jpg', './samples/sample5.jpg', './samples/sample6.jpg']

for sample in samples:
    if not os.path.exists(sample):
        print(f"Warning: {sample} not found. Skipping.")
        continue
        
    try:
        broken, worn, inner_status, visualization = inspector.inspect_gear(sample)
        print(f"{sample}: Broken teeth: {broken}, Worn teeth: {worn}, Inner diameter: {inner_status}")
        
        cv2.imshow(sample, visualization)
        
    except Exception as e:
        print(f"Error processing {sample}: {str(e)}")

# Wait for key press and clean up
cv2.waitKey(0)
cv2.destroyAllWindows()

# hours wasted on bonus: 4