from baseballcv.functions import LoadTools, DataTools
from baseballcv.models import Florence2
import cv2
import numpy as np
from datetime import datetime

class BatTrackingPipeline:
    """
    A specialized pipeline for analyzing baseball plays through bat tracking
    and swing analysis
    """
    def __init__(self):
        # Initialize core tools
        self.load_tools = LoadTools()
        self.data_tools = DataTools()
        
        # Load specialized models
        self.bat_model = self.load_model_with_retry("bat_tracking")
        self.context_model = Florence2()
        
        # Set up logging and monitoring
        self.setup_logging()
    
    def load_model_with_retry(self, model_name, max_attempts=3):
        """
        Robust model loading with retry logic
        """
        for attempt in range(max_attempts):
            try:
                model_path = self.load_tools.load_model(model_name)
                return YOLO(model_path)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise Exception(f"Failed to load model after {max_attempts} attempts")
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
    
    def process_sequence(self, video_path):
        """
        Process a complete baseball swing sequence
        """
        # Extract frames from the video
        self.logger.info("Extracting frames from video sequence")
        frames = self.data_tools.generate_photo_dataset(
            video_path,
            max_num_frames=60,  # Capture full swing sequence
            output_frames_folder="swing_analysis_frames"
        )
        
        # Initialize analysis containers
        sequence_data = []
        game_context = None
        
        # Process each frame in the sequence
        self.logger.info("Beginning frame analysis")
        for idx, frame in enumerate(frames):
            # Track the bat
            bat_results = self.bat_model.predict(
                frame,
                conf=0.30,  # Higher confidence threshold for bat detection
                verbose=False
            )
            
            # Get game context from the first frame
            if idx == 0:
                self.logger.info("Analyzing game context")
                game_context = self.context_model.inference(
                    frame,
                    task="<BATTING_CONTEXT>"
                )
            
            # Combine results for this frame
            frame_data = self.process_frame_results(bat_results)
            sequence_data.append(frame_data)
        
        # Analyze the complete sequence
        return self.analyze_sequence(sequence_data, game_context)
    
    def process_frame_results(self, bat_results):
        """
        Extract and organize bat detection results from a single frame
        """
        frame_data = {
            'timestamp': bat_results.timestamp,
            'bat_detections': [],
            'bat_angle': None  # Added for swing analysis
        }
        
        # Process each detection in the frame
        for detection in bat_results.boxes:
            if detection.conf[0] > 0.30:  # Confidence threshold
                bat_box = detection.xyxy[0].tolist()
                frame_data['bat_detections'].append({
                    'position': bat_box,
                    'confidence': detection.conf[0].item(),
                    'angle': self.calculate_bat_angle(bat_box)  # Calculate bat angle
                })
        
        return frame_data
    
    def calculate_bat_angle(self, bat_box):
        """
        Calculate the approximate angle of the bat based on its bounding box
        """
        x1, y1, x2, y2 = bat_box
        angle = np.arctan2(y2 - y1, x2 - x1)
        return np.degrees(angle)
    
    def analyze_sequence(self, sequence_data, game_context):
        """
        Analyze the complete sequence of frames to understand the swing
        """
        # Filter and clean bat positions and angles
        bat_trajectory = self.extract_clean_trajectories(sequence_data)
        
        # Calculate swing metrics
        swing_metrics = self.calculate_swing_metrics(bat_trajectory)
        
        # Organize findings
        analysis_results = {
            'sequence_length': len(sequence_data),
            'bat_trajectory': bat_trajectory,
            'swing_metrics': swing_metrics,
            'game_situation': game_context,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis_results
    
    def extract_clean_trajectories(self, sequence_data):
        """
        Clean and organize bat position and angle data across frames
        """
        trajectories = []
        
        for frame_data in sequence_data:
            if frame_data['bat_detections']:
                # Take the highest confidence detection
                best_detection = max(
                    frame_data['bat_detections'],
                    key=lambda x: x['confidence']
                )
                trajectories.append({
                    'position': best_detection['position'],
                    'angle': best_detection['angle']
                })
            else:
                trajectories.append(None)
        
        # Interpolate missing positions and angles
        return self.interpolate_missing_trajectories(trajectories)
    
    def calculate_swing_metrics(self, bat_trajectory):
        """
        Calculate key metrics about the swing
        """
        metrics = {
            'max_bat_speed': 0,
            'swing_plane_angle': 0,
            'contact_point': None,
            'swing_duration': 0
        }
        
        # Calculate metrics from trajectory data
        if bat_trajectory:
            # Implementation of swing metrics calculations
            # This would include bat speed, swing plane, etc.
            pass
        
        return metrics

# Using the pipeline
def analyze_baseball_swing(video_path):
    """
    Analyze a baseball swing using our custom pipeline
    """
    pipeline = BatTrackingPipeline()
    
    try:
        results = pipeline.process_sequence(video_path)
        print(f"Successfully analyzed swing sequence of {results['sequence_length']} frames")
        return results
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    results = analyze_baseball_swing("swing_sequence.mp4") 