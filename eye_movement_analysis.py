import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import math
import logging
from typing import Dict, Any, List
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices for MediaPipe
LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

class EyeMovementAnalyzer:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def analyze_video(self, video_bytes: bytes, test_metadata: dict = None) -> Dict[str, Any]:
        """Analyze eye movements in the uploaded video"""
        temp_video_path = None
        
        try:
            # Save video bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as temp_file:
                temp_file.write(video_bytes)
                temp_video_path = temp_file.name
            
            logger.info(f"Processing eye movement video: {len(video_bytes)} bytes")
            
            cap = cv2.VideoCapture(temp_video_path)
            
            if not cap.isOpened():
                logger.error("Failed to open video file for eye movement analysis")
                return self.create_empty_result()
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Eye movement video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
            
            eye_movements = []
            frame_count = 0
            
            # Process every 3rd frame for performance
            sample_interval = 3
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sample_interval == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    results = self.face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # Extract eye positions
                            eye_data = self.extract_eye_positions(face_landmarks, frame.shape)
                            eye_data['timestamp'] = timestamp
                            eye_data['frame'] = frame_count
                            eye_movements.append(eye_data)
                            break  # Only process first face
                
                frame_count += 1
            
            cap.release()
            
            # Analyze the collected data
            analysis_results = self.analyze_eye_movement_patterns(eye_movements, test_metadata)
            
            logger.info(f"Eye movement analysis complete: {len(eye_movements)} data points")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in eye movement analysis: {str(e)}")
            return self.create_empty_result()
            
        finally:
            # Clean up temporary file
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass
    
    def extract_eye_positions(self, face_landmarks, frame_shape):
        """Extract eye center positions and other metrics"""
        h, w = frame_shape[:2]
        
        # Get left eye landmarks
        left_eye_points = []
        for idx in LEFT_EYE_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            left_eye_points.append([landmark.x * w, landmark.y * h])
        
        # Get right eye landmarks  
        right_eye_points = []
        for idx in RIGHT_EYE_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            right_eye_points.append([landmark.x * w, landmark.y * h])
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        
        # Calculate gaze direction (simplified)
        gaze_vector = self.estimate_gaze_direction(left_eye_points, right_eye_points)
        
        # Calculate eye openness
        left_openness = self.calculate_eye_openness(left_eye_points)
        right_openness = self.calculate_eye_openness(right_eye_points)
        
        return {
            'left_eye_center': left_eye_center.tolist(),
            'right_eye_center': right_eye_center.tolist(),
            'gaze_vector': gaze_vector,
            'left_openness': left_openness,
            'right_openness': right_openness,
            'average_eye_center': ((left_eye_center + right_eye_center) / 2).tolist()
        }
    
    def estimate_gaze_direction(self, left_eye_points, right_eye_points):
        """Estimate gaze direction based on eye shape"""
        # Simplified gaze estimation
        left_eye_array = np.array(left_eye_points)
        right_eye_array = np.array(right_eye_points)
        
        # Calculate horizontal gaze by looking at eye shape asymmetry
        left_width = np.max(left_eye_array[:, 0]) - np.min(left_eye_array[:, 0])
        right_width = np.max(right_eye_array[:, 0]) - np.min(right_eye_array[:, 0])
        
        # Simple horizontal gaze estimation
        horizontal_gaze = (left_width - right_width) / max(left_width, right_width, 1)
        
        return {
            'horizontal': float(horizontal_gaze),
            'vertical': 0.0,  # Simplified
            'magnitude': abs(horizontal_gaze)
        }
    
    def calculate_eye_openness(self, eye_points):
        """Calculate how open an eye is"""
        eye_array = np.array(eye_points)
        
        # Calculate height vs width ratio
        height = np.max(eye_array[:, 1]) - np.min(eye_array[:, 1])
        width = np.max(eye_array[:, 0]) - np.min(eye_array[:, 0])
        
        return float(height / max(width, 1))
    
    def analyze_eye_movement_patterns(self, eye_movements, test_metadata):
        """Analyze the collected eye movement data"""
        if not eye_movements:
            return self.create_empty_result()
        
        # Calculate movement smoothness
        smoothness_score = self.calculate_movement_smoothness(eye_movements)
        
        # Calculate accuracy (how well they followed H-pattern)
        accuracy_score = self.calculate_h_pattern_accuracy(eye_movements)
        
        # Calculate completion rate
        completion_rate = min(len(eye_movements) / 80, 1.0)  # Assuming ~8 seconds at 10fps
        
        # Generate movement heatmap data
        heatmap_data = self.generate_movement_heatmap(eye_movements)
        
        return {
            'success': True,
            'analysis_complete': True,
            'timestamp': datetime.now().isoformat(),
            'video_info': {
                'total_frames_analyzed': len(eye_movements),
                'sample_rate': 'Every 3rd frame'
            },
            'scores': {
                'smoothness_score': round(smoothness_score, 3),
                'accuracy_score': round(accuracy_score, 3),
                'completion_rate': round(completion_rate, 3)
            },
            'detailed_metrics': {
                'average_gaze_magnitude': round(np.mean([em['gaze_vector']['magnitude'] for em in eye_movements]), 3),
                'max_horizontal_movement': round(max([abs(em['gaze_vector']['horizontal']) for em in eye_movements]), 3),
                'eye_coordination': round(self.calculate_eye_coordination(eye_movements), 3),
                'movement_consistency': round(self.calculate_movement_consistency(eye_movements), 3)
            },
            'heatmap_data': heatmap_data,
            'recommendations': self.generate_recommendations(smoothness_score, accuracy_score, completion_rate)
        }
    
    def calculate_movement_smoothness(self, eye_movements):
        """Calculate how smooth the eye movements are"""
        if len(eye_movements) < 3:
            return 0.0
        
        # Get average eye center positions
        positions = [em['average_eye_center'] for em in eye_movements]
        
        # Calculate velocity changes
        velocities = []
        for i in range(1, len(positions)):
            velocity = math.sqrt(
                (positions[i][0] - positions[i-1][0])**2 + 
                (positions[i][1] - positions[i-1][1])**2
            )
            velocities.append(velocity)
        
        # Calculate acceleration changes
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = abs(velocities[i] - velocities[i-1])
            accelerations.append(acceleration)
        
        if not accelerations:
            return 0.5
        
        # Smooth movements have lower acceleration variance
        smoothness = 1.0 - min(np.std(accelerations) / max(np.mean(accelerations), 1), 1.0)
        
        return max(0.0, min(1.0, smoothness))
    
    def calculate_h_pattern_accuracy(self, eye_movements):
        """Calculate how accurately the H-pattern was followed"""
        if len(eye_movements) < 8:
            return 0.0
        
        positions = [em['average_eye_center'] for em in eye_movements]
        
        # Normalize positions to 0-1 range
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        if len(set(x_coords)) < 2 or len(set(y_coords)) < 2:
            return 0.0
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Check for expected movements (left/right/up/down ranges)
        left_movements = sum(1 for x in x_coords if x < (x_min + (x_max - x_min) * 0.3))
        right_movements = sum(1 for x in x_coords if x > (x_min + (x_max - x_min) * 0.7))
        up_movements = sum(1 for y in y_coords if y < (y_min + (y_max - y_min) * 0.3))
        down_movements = sum(1 for y in y_coords if y > (y_min + (y_max - y_min) * 0.7))
        
        # Score based on coverage of all directions
        direction_coverage = sum([
            1 if left_movements > 0 else 0,
            1 if right_movements > 0 else 0,
            1 if up_movements > 0 else 0,
            1 if down_movements > 0 else 0
        ]) / 4.0
        
        return direction_coverage
    
    def generate_movement_heatmap(self, eye_movements):
        """Generate heatmap data for visualization"""
        positions = [em['average_eye_center'] for em in eye_movements]
        
        if not positions:
            return []
        
        # Create a simplified heatmap by binning positions
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Normalize to 0-1 range
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        heatmap_data = []
        grid_size = 8  # Smaller grid for performance
        
        for i in range(grid_size):
            for j in range(grid_size):
                x_bin = x_min + (x_max - x_min) * i / grid_size if x_max > x_min else x_min
                y_bin = y_min + (y_max - y_min) * j / grid_size if y_max > y_min else y_min
                
                # Count positions in this bin
                count = sum(1 for pos in positions 
                           if x_bin <= pos[0] < x_bin + (x_max - x_min) / grid_size
                           and y_bin <= pos[1] < y_bin + (y_max - y_min) / grid_size)
                
                if count > 0:
                    heatmap_data.append({
                        'x': round(i / grid_size, 2),
                        'y': round(j / grid_size, 2),
                        'intensity': round(count / len(positions), 3)
                    })
        
        return heatmap_data
    
    def calculate_eye_coordination(self, eye_movements):
        """Calculate how well left and right eyes coordinate"""
        if not eye_movements:
            return 0.0
        
        coordination_scores = []
        
        for em in eye_movements:
            # Calculate openness difference
            openness_diff = abs(em['left_openness'] - em['right_openness'])
            
            # Good coordination = similar openness
            coordination = 1.0 - min(openness_diff, 1.0)
            coordination_scores.append(coordination)
        
        return np.mean(coordination_scores)
    
    def calculate_movement_consistency(self, eye_movements):
        """Calculate consistency of movements"""
        if len(eye_movements) < 10:
            return 0.5
        
        positions = [em['average_eye_center'] for em in eye_movements]
        
        # Calculate velocity consistency
        velocities = []
        for i in range(1, len(positions)):
            velocity = math.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                               (positions[i][1] - positions[i-1][1])**2)
            velocities.append(velocity)
        
        if not velocities:
            return 0.5
        
        # Consistency = low variance in velocity
        velocity_std = np.std(velocities)
        velocity_mean = np.mean(velocities)
        
        consistency = 1.0 - min(velocity_std / max(velocity_mean, 1), 1.0)
        return max(0.0, consistency)
    
    def generate_recommendations(self, smoothness, accuracy, completion):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if completion < 0.8:
            recommendations.append("Consider retaking the test in better lighting conditions")
        
        if smoothness < 0.7:
            recommendations.append("Practice smooth, controlled eye movements without head motion")
        
        if accuracy < 0.7:
            recommendations.append("Focus on following directions precisely and maintaining steady gaze")
        
        if completion >= 0.9 and smoothness >= 0.8:
            recommendations.append("Excellent eye movement control! Results are within normal range")
        
        recommendations.append("Consult with a healthcare professional for detailed interpretation")
        
        return recommendations
    
    def create_empty_result(self):
        """Create empty result when analysis fails"""
        return {
            'success': False,
            'error': 'No eye movements detected or analysis failed',
            'scores': {
                'smoothness_score': 0.0,
                'accuracy_score': 0.0,
                'completion_rate': 0.0
            },
            'detailed_metrics': {
                'average_gaze_magnitude': 0.0,
                'max_horizontal_movement': 0.0,
                'eye_coordination': 0.0,
                'movement_consistency': 0.0
            },
            'heatmap_data': [],
            'recommendations': ['Unable to analyze eye movements. Please ensure good lighting and stable camera position.']
        }