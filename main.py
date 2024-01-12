import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def detect_pose(image, pose):
    results = pose.process(image)
    return results.pose_landmarks

def draw_landmark(frame, landmark, color=(0, 255, 0)):
    if landmark:
        height, width, _ = frame.shape
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(frame, (cx, cy), 5, color, -1)  # Draw a circle at the landmark

def calculate_distance_ratio(landmarks, nose_landmark, shoulder_landmarks):
    if landmarks and nose_landmark and shoulder_landmarks:
        shoulder_span_distance = np.linalg.norm([shoulder_landmarks[0].x - shoulder_landmarks[1].x, shoulder_landmarks[0].y - shoulder_landmarks[1].y])
        distance_ratios = []

        for landmark in landmarks.landmark:
            # Calculate the distance from the landmark to the nose
            distance_nose = np.linalg.norm([landmark.x - nose_landmark.x, landmark.y - nose_landmark.y])

            # Calculate the distance ratio using shoulder span distance as the denominator
            ratio = distance_nose / shoulder_span_distance
            distance_ratios.append(ratio)

        return distance_ratios
    else:
        return None

def calculate_similarity_score(ratio_live, ratio_reference):
    # Calculate the similarity score based on the absolute difference between live and reference ratios
    return 1 - abs(ratio_live - ratio_reference)

def color_gradient(score):
    # Generate a gradient color based on the score
    red = int(max(255 * (1 - score), 0))
    green = int(max(255 * score, 0))
    blue = 0
    return (blue, green, red)

def compare_poses(video_path, use_live_camera=True):
    cap_reference = cv2.VideoCapture(video_path)
    
    # Initialize live camera if specified
    if use_live_camera:
        cap_live = cv2.VideoCapture(0)  # Use default camera (you can change the index if needed)
    else:
        cap_live = cv2.VideoCapture(video_path)

    reference_landmarks = None  # Store reference landmarks from the first frame of the reference video

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap_reference.isOpened() and cap_live.isOpened():
            ret_reference, frame_reference = cap_reference.read()
            
            # Capture frame from live camera or video file based on the flag
            if use_live_camera:
                ret_live, frame_live = cap_live.read()
            else:
                ret_live, frame_live = cap_live.read()

            if not ret_reference or not ret_live:
                print("Error reading frames.")
                break

            landmarks_reference = detect_pose(frame_reference, pose)
            landmarks_live = detect_pose(frame_live, pose)

            if landmarks_reference is None or landmarks_live is None:
                # If either landmarks are not detected, skip to the next frame
                continue

            if reference_landmarks is None:
                # Store reference landmarks from the first frame of the reference video
                reference_landmarks = landmarks_reference

            total_score = 0  # Initialize total score for all landmarks

            for i, landmark in enumerate(landmarks_reference.landmark):
                # Draw each reference landmark in green
                draw_landmark(frame_reference, landmark, color=(0, 255, 0))

            for i, landmark in enumerate(landmarks_live.landmark):
                # Extract nose and shoulder landmarks for the live video
                nose_landmark_live = landmarks_live.landmark[mp_pose.PoseLandmark.NOSE]
                shoulder_landmarks_live = [landmarks_live.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks_live.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]]

                # Extract nose and shoulder landmarks for the reference video
                nose_landmark_ref = reference_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                shoulder_landmarks_ref = [reference_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER], reference_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]]

                # Calculate distance ratio for the live landmark
                distance_ratio_live = calculate_distance_ratio(landmarks_live, nose_landmark_live, shoulder_landmarks_live)[i]

                # Calculate distance ratio for the reference landmark
                distance_ratio_reference = calculate_distance_ratio(reference_landmarks, nose_landmark_ref, shoulder_landmarks_ref)[i]

                # Calculate the similarity score for the landmark
                score = calculate_similarity_score(distance_ratio_live, distance_ratio_reference)

                total_score += score  # Add individual scores to the total

                # Get color based on the score
                landmark_color = color_gradient(score)

                # Draw the live landmark with gradient color
                draw_landmark(frame_live, landmark, color=landmark_color) 

            # Calculate overall score as the average of individual scores
            overall_score = total_score / len(landmarks_reference.landmark)

            # Display the overall score on the top right corner of the live video
            cv2.putText(frame_live, f"Overall Score: {overall_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Reference Video", frame_reference)
            cv2.imshow("Live Video", frame_live)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting the loop.")
                break

    cap_reference.release()
    cap_live.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "source_video/ornekvideo.mp4"
    compare_poses(video_path, use_live_camera=True)
