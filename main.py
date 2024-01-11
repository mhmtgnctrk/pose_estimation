import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

mp_pose = mp.solutions.pose

def detect_pose(image, pose):
    with pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detector:
        results = pose_detector.process(image)
        return results.pose_landmarks

def compare_poses(video_path, use_live_camera=True):
    cap1 = cv2.VideoCapture(video_path)
    
    # Initialize live camera if specified
    if use_live_camera:
        cap2 = cv2.VideoCapture(0)  # Use default camera (you can change the index if needed)
    else:
        cap2 = cv2.VideoCapture(video_path)

    with mp_pose.Pose() as pose:
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            
            # Capture frame from live camera or video file based on the flag
            if use_live_camera:
                ret2, frame2 = cap2.read()
            else:
                ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            landmarks1 = detect_pose(frame1, pose)
            landmarks2 = detect_pose(frame2, pose)

            if landmarks1 and landmarks2:
                # Compare the pose landmarks or perform further analysis
                # For simplicity, let's just print the first landmark's x-coordinate
                x1 = landmarks1.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                x2 = landmarks2.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x

                print(f"Left shoulder x-coordinate in video 1: {x1}")
                print(f"Left shoulder x-coordinate in video 2: {x2}")
                print()

            cv2.imshow("Video 1", frame1)
            cv2.imshow("Video 2", frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "path/to/video.mp4"
    compare_poses(video_path, use_live_camera=True)
