import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

class SpeedLimitDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the speed limit detector
        model_path: Path to YOLO model (you'll need to train/fine-tune for speed signs)
        """
        self.model = YOLO(model_path)
        self.detection_history = deque(maxlen=30)  # Track last 30 frames
        
        # Define sign types and their colors for visualization
        self.sign_colors = {
            'speed_limit': (0, 255, 0),      # Green
            'construction': (0, 165, 255),   # Orange
            'school_zone': (0, 255, 255)     # Yellow
        }
        
        # OCR-like digit templates (simplified - for production use Tesseract/EasyOCR)
        self.speed_values = []
        
    def preprocess_roi(self, roi):
        """Preprocess region of interest for better detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def detect_sign_type(self, roi):
        """
        Detect if sign is construction, school zone, or regular speed limit
        Uses color analysis and contextual features
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        # Orange (construction) - HSV ranges
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Yellow (school zone) - HSV ranges
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Calculate percentages
        total_pixels = roi.shape[0] * roi.shape[1]
        orange_pct = (cv2.countNonZero(orange_mask) / total_pixels) * 100
        yellow_pct = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
        
        # Classification logic
        if orange_pct > 15:
            return 'construction'
        elif yellow_pct > 20:
            return 'school_zone'
        else:
            return 'speed_limit'
    
    def extract_speed_value(self, roi):
        """
        Extract speed limit number from sign
        For production, integrate Tesseract OCR or EasyOCR
        """
        # Preprocess
        processed = self.preprocess_roi(roi)
        
        # Simple template matching or OCR would go here
        # For now, return placeholder - you'll integrate OCR
        # Example with pytesseract:
        # import pytesseract
        # text = pytesseract.image_to_string(processed, config='--psm 7 digits')
        # speed = ''.join(filter(str.isdigit, text))
        
        return "??"  # Placeholder
    
    def process_frame(self, frame):
        """Process a single frame and detect speed limit signs"""
        results = self.model(frame, conf=0.5)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Extract ROI
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Classify sign type
                sign_type = self.detect_sign_type(roi)
                
                # Extract speed value
                speed_value = self.extract_speed_value(roi)
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'sign_type': sign_type,
                    'speed_value': speed_value
                }
                
                detections.append(detection)
                
                # Draw on frame
                color = self.sign_colors.get(sign_type, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                label = f"{sign_type.replace('_', ' ').title()}: {speed_value} mph"
                label_y = y1 - 10 if y1 > 30 else y1 + 20
                
                # Draw label background
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame, (x1, label_y - text_h - 5), 
                    (x1 + text_w, label_y + 5), color, -1
                )
                
                # Draw label text
                cv2.putText(
                    frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                )
        
        return frame, detections
    
    def process_video(self, video_path, output_path=None, display=True):
        """Process video file or webcam stream"""
        # Open video
        if video_path == 0:
            cap = cv2.VideoCapture(0)  # Webcam
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("Processing video... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame, detections = self.process_frame(frame)
            
            # Add FPS counter
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                processed_frame, f"FPS: {current_fps:.1f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # Display
            if display:
                cv2.imshow('Speed Limit Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write to output
            if writer:
                writer.write(processed_frame)
            
            # Print detections
            if detections:
                print(f"Frame {frame_count}: {len(detections)} signs detected")
                for det in detections:
                    print(f"  - {det['sign_type']}: {det['speed_value']} mph "
                          f"(conf: {det['confidence']:.2f})")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {frame_count} frames in {elapsed:.2f}s")
        print(f"Average FPS: {current_fps:.2f}")


def main():
    """Example usage"""
    # Initialize detector
    # Note: You'll need to train or fine-tune YOLO on speed limit signs
    detector = SpeedLimitDetector('yolov8n.pt')
    
    # Process webcam (use 0 for default webcam)
    # detector.process_video(0, display=True)
    
    # Process video file
    # detector.process_video('input_video.mp4', 'output_video.mp4', display=True)
    
    # For testing, process webcam
    print("Starting webcam detection...")
    print("Make sure you have a trained YOLO model for speed signs!")
    detector.process_video(0, display=True)


if __name__ == "__main__":
    main()