import argparse
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
from utils_data_process import get_hand_joints_mano_single_hand, calculate_signed_angle_between_vectors, calculate_signed_angle_between_planes
from common.pose_utils import pose_to_mat, mat_to_pose


class HandDetector:
    """Hand keypoint detector using MediaPipe Hands"""
    
    def __init__(self, static_image_mode=False, max_num_hands=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect(self, image):
        """Detect hand keypoints
        
        Args:
            image: BGR format image
            
        Returns:
            landmarks: 21 keypoints 3D coordinates (21, 3), returns None if not detected
            hand_side: 'Left' or 'Right', returns None if not detected
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0]
            hand_side = handedness.classification[0].label  # 'Left' or 'Right'
            
            # 提取21个关键点的3D坐标
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
            ])
            
            # Convert normalized coordinates to pixel coordinates (x, y) and relative depth (z)
            h, w = image.shape[:2]
            landmarks[:, 0] *= w  # x coordinate
            landmarks[:, 1] *= h  # y coordinate
            # z coordinate remains normalized, but we need to estimate real 3D depth
            
            return landmarks, hand_side
        
        return None, None
    
    def draw_landmarks(self, image, landmarks):
        """Draw keypoints on image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        
        return image


def estimate_3d_from_2d_landmarks(landmarks_2d, image_shape, front_camera=False):
    """Estimate 3D coordinates from 2D keypoints
    
    Uses simple depth estimation method: estimates depth based on geometric relationships of hand keypoints
    
    Args:
        landmarks_2d: (21, 3) 2D keypoints with normalized z
        image_shape: (h, w) or (h, w, c) image shape
        front_camera: If True, flip x coordinate to account for front camera mirroring
    """
    h, w = image_shape[:2]
    
    # Normalize to [-1, 1] range
    landmarks_normalized = landmarks_2d.copy()
    landmarks_normalized[:, 0] = (landmarks_normalized[:, 0] - w / 2) / (w / 2)
    landmarks_normalized[:, 1] = (landmarks_normalized[:, 1] - h / 2) / (h / 2)
    
    # If front camera, flip x coordinate to account for mirroring
    if front_camera:
        landmarks_normalized[:, 0] = -landmarks_normalized[:, 0]
    
    # Use hand size to estimate depth (assume hand is about 0.5m from camera)
    # Estimate hand size (using distance from wrist to middle finger MCP)
    wrist = landmarks_2d[0]  # wrist
    middle_mcp = landmarks_2d[9]  # middle finger MCP
    
    hand_size_2d = np.linalg.norm(wrist[:2] - middle_mcp[:2])
    # Assume actual hand size is about 0.1m, estimate depth based on 2D size scaling
    estimated_depth = 0.1 * w / hand_size_2d if hand_size_2d > 0 else 0.5
    
    # Convert normalized coordinates to 3D coordinates (unit: meters)
    landmarks_3d = np.zeros((21, 3))
    landmarks_3d[:, 0] = landmarks_normalized[:, 0] * estimated_depth * 0.5  # x (meters)
    landmarks_3d[:, 1] = -landmarks_normalized[:, 1] * estimated_depth * 0.5  # y (meters, flipped)
    landmarks_3d[:, 2] = estimated_depth + landmarks_2d[:, 2] * 0.05  # z (meters, using relative depth)
    
    return landmarks_3d


def convert_landmarks_to_poses(landmarks_3d):
    """Convert keypoints to 6D pose format
    
    Note: Function expects hand_pose to contain 20 keypoints (indices 1-20, excluding wrist 0),
    plus wrist_pose, totaling 21 keypoints.
    
    Args:
        landmarks_3d: (21, 3) 3D coordinates of keypoints, index 0 is wrist
        
    Returns:
        hand_pose: (20, 6) 6D pose of 20 keypoints (in world coordinate system, not relative to wrist)
        wrist_pose: (6,) 6D pose of wrist (in world coordinate system)
    """
    # Wrist position and rotation
    wrist_pos = landmarks_3d[0].copy()
    
    # Build wrist coordinate system
    # Use vectors from wrist to middle finger MCP and index finger MCP to build coordinate system
    middle_mcp = landmarks_3d[9]
    index_mcp = landmarks_3d[5]
    
    # X-axis: points to middle finger direction
    x_axis = middle_mcp - wrist_pos
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
    # Y-axis: points to index finger direction (in palm plane)
    y_temp = index_mcp - wrist_pos
    y_axis = y_temp - np.dot(y_temp, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # Z-axis: perpendicular to palm plane
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    
    # Build wrist rotation matrix
    wrist_rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    wrist_rotvec = R.from_matrix(wrist_rot).as_rotvec()
    wrist_pose = np.concatenate([wrist_pos, wrist_rotvec])
    
    # Calculate pose for each keypoint (in world coordinate system, not relative to wrist)
    # Function will convert them to wrist coordinate system internally
    hand_pose = np.zeros((20, 6))
    for i in range(20):
        # Use absolute position of keypoint
        pos = landmarks_3d[i + 1].copy()
        # Use zero rotation (or can estimate based on keypoint direction)
        # Function will calculate transformation relative to wrist internally
        hand_pose[i] = np.concatenate([pos, np.zeros(3)])
    
    return hand_pose, wrist_pose


def draw_joint_angles(image, joint_angles_dict, x_offset=10, y_offset=30):
    """Draw joint angles on image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)
    thickness = 1
    line_height = 20
    
    y = y_offset
    for joint_name, angle in joint_angles_dict.items():
        text = f"{joint_name}: {angle:.3f}"
        cv2.putText(image, text, (x_offset, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height
    
    return image


def get_joint_angles_dict(joint_angles):
    """Convert array values in joint angles dictionary to scalars"""
    # Function already returns dictionary, just need to handle possible array values
    result = {}
    for key, value in joint_angles.items():
        if isinstance(value, np.ndarray):
            # If array, take first element (because T=1)
            if value.size == 1:
                result[key] = float(value.item())
            else:
                result[key] = float(value.flatten()[0])
        else:
            result[key] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser(description='Real-time MANO hand keypoint detection and joint angle calculation')
    parser.add_argument('--hand', type=str, choices=['left', 'right'], default='right',
                       help='Detect left or right hand (default: right)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--width', type=int, default=1080, help='Camera width (default: 1080)')
    parser.add_argument('--height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--front_camera', action='store_true',
                       help='Use front camera mode (automatically enables mirror flip and hand detection reversal)')
    
    args = parser.parse_args()
    
    is_left = (args.hand == 'left')
    
    # Initialize hand detector
    print("Initializing MediaPipe Hands...")
    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("MediaPipe Hands initialization complete!")
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return
    
    print(f"Starting real-time detection ({'left' if is_left else 'right'} hand)...")
    if args.front_camera:
        print("📷 Front camera mode enabled (auto-enables mirror flip and hand detection reversal)")
    print("Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera frame")
            break
        
        # # If front camera, flip image horizontally
        # if args.front_camera:
        #     frame = cv2.flip(frame, 1)  # 1 means horizontal flip
        
        frame_count += 1
        
        # Detect hand keypoints
        landmarks_2d, detected_hand_side = detector.detect(frame)
        
        if landmarks_2d is not None:
            # Process left/right hand detection result
            if args.front_camera:
                # Front camera mode: reverse MediaPipe detection result
                detected_hand_side = 'Right' if detected_hand_side == 'Left' else 'Left'
            
            detected_is_left = (detected_hand_side == 'Left')
            
            if detected_is_left == is_left:
                
                # Estimate 3D coordinates
                landmarks_3d = estimate_3d_from_2d_landmarks(landmarks_2d, frame.shape, front_camera=args.front_camera)
                
                # Convert to pose format
                hand_pose, wrist_pose = convert_landmarks_to_poses(landmarks_3d)
                
                # Expand dimensions to match function requirement (T, N, 6)
                hand_pose = hand_pose[None, :, :]  # (1, 20, 6)
                wrist_pose = wrist_pose[None, :]   # (1, 6)
                
                try:
                    # Calculate joint angles
                    joint_angles, kps = get_hand_joints_mano_single_hand(
                        hand_pose, wrist_pose, is_left=is_left
                    )
                    
                    # Draw keypoints
                    frame = detector.draw_landmarks(frame, landmarks_2d)
                    
                    # Draw joint angles (function already returns dictionary format)
                    joint_angles_dict = get_joint_angles_dict(joint_angles)
                    # Display all joint angles
                    limited_dict = dict(list(joint_angles_dict.items()))
                    frame = draw_joint_angles(frame, limited_dict)
                    
                    # Display hand information
                    hand_text = f"Detected: {detected_hand_side} hand"
                    cv2.putText(frame, hand_text, (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                except Exception as e:
                    print(f"Error calculating joint angles: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Detected hand does not match parameter
                text = f"Please show {'left' if is_left else 'right'} hand! Currently detected: {detected_hand_side} hand"
                cv2.putText(frame, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No hand detected
            text = f"No {'left' if is_left else 'right'} hand detected, please place your hand in front of camera"
            cv2.putText(frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display FPS
        fps_text = f"FPS: {frame_count}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('MANO Hand Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited")


if __name__ == '__main__':
    main()
