import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

def detect_pose(image, pose):
    results = pose.process(image)
    return results.pose_landmarks

def draw_landmark(frame, x, y, color=(0, 255, 0)):
    if x and y:
        height, width, _ = frame.shape
        cx, cy = int(x * width), int(y * height)
        cv2.circle(frame, (cx, cy), 5, color, -1)  # Draw a circle at the landmark

def apply_ema_filter(current_landmarks, previous_landmarks, alpha=0.35):
    if previous_landmarks is None:
        return current_landmarks

    filtered_landmarks = []

    for current, previous in zip(current_landmarks, previous_landmarks):
        current_x, current_y = current
        previous_x, previous_y = previous

        # Apply EMA filter to x and y coordinates separately
        filtered_x = alpha * current_x + (1 - alpha) * previous_x
        filtered_y = alpha * current_y + (1 - alpha) * previous_y

        filtered_landmarks.append((filtered_x, filtered_y))


    return filtered_landmarks

def calculate_distance_ratio(landmarks, body_center_landmark, shoulder_landmarks):
    if landmarks and body_center_landmark and shoulder_landmarks:
        shoulder_span_distance = np.linalg.norm([shoulder_landmarks[0][0] - shoulder_landmarks[1][0], shoulder_landmarks[0][1] - shoulder_landmarks[1][1]])
        distance_ratios = []

        for landmark in landmarks:
            # Calculate the distance from the landmark to the body center
            distance_body_center = np.linalg.norm([landmark[0] - body_center_landmark[0], landmark[1] - body_center_landmark[1]])

            # Calculate the distance ratio using shoulder span distance
            ratio = distance_body_center / shoulder_span_distance
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
    green = int(max(255 * score ** 2 , 0))
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
    previous_live_landmarks = None
    previous_ref_landmarks = None
    overall_scores=[]
    timestamps=[]

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.4) as pose:
        while cap_reference.isOpened() and cap_live.isOpened():
            ret_reference, frame_reference = cap_reference.read()
            
            # Capture frame from live camera or video file based on the flag
            if use_live_camera:
                ret_live, frame_live = cap_live.read()
            else:
                ret_live, frame_live = cap_reference.read()

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
                
            #Apply EMA fılter
            landmarks_live_smoothed = apply_ema_filter(
                [(landmark.x, landmark.y) for landmark in landmarks_live.landmark],
                previous_live_landmarks
            )
            reference_landmarks_smoothed = apply_ema_filter(
                [(landmark.x, landmark.y) for landmark in landmarks_reference.landmark],
                previous_ref_landmarks
            )

            total_score = 0  # Initialize total score for all landmarks

            #for i, landmark in enumerate(reference_landmarks_smoothed):
                # Draw each reference landmark in green
            #    draw_landmark(frame_reference, landmark[0], landmark[1], color=(0, 255, 0))

            # Extract body_center and shoulder landmarks for the live video
            body_center_landmark_live = [landmarks_live_smoothed[0][0], 
                                    landmarks_live_smoothed[24][1]]
            shoulder_landmarks_live = [[landmarks_live_smoothed[12][0], landmarks_live_smoothed[12][1]],
                                        [landmarks_live_smoothed[11][0], landmarks_live_smoothed[11][1]]]

            # Extract body_center and shoulder landmarks for the reference video
            body_center_landmark_ref = [reference_landmarks_smoothed[0][0], 
                                    reference_landmarks_smoothed[24][1]]
            shoulder_landmarks_ref = [[reference_landmarks_smoothed[12][0], reference_landmarks_smoothed[12][1]],
                                        [reference_landmarks_smoothed[11][0], reference_landmarks_smoothed[11][1]]]

            # Calculate distance ratio for the live landmark
            distance_ratios_live = calculate_distance_ratio(landmarks_live_smoothed, body_center_landmark_live, shoulder_landmarks_live)
            
            # Calculate distance ratio for the reference landmark
            distance_ratios_reference = calculate_distance_ratio(reference_landmarks_smoothed, body_center_landmark_ref, shoulder_landmarks_ref)
            
            for i, landmark in enumerate(landmarks_live_smoothed):

                # Calculate the similarity score for the landmark
                score = calculate_similarity_score(distance_ratios_live[i], distance_ratios_reference[i])

                total_score += score  # Add individual scores to the total

                # Get color # In the given code, there is no variable `b` or any reference to it.
                landmark_color = color_gradient(score)

                # Draw the live landmark with gradient color
                draw_landmark(frame_live, landmark[0], landmark[1], color=landmark_color) 

            # Calculate overall score as the average of individual scores
            overall_score = total_score / len(landmarks_live_smoothed)
            #Append the scores to array every 0.5 seconds
            if overall_score >= 0:
                if len(timestamps) > 0 and cap_reference.get(cv2.CAP_PROP_POS_MSEC)*1e-3 - timestamps[-1] > 0.5:
                    overall_scores.append(overall_score*100)  
                    timestamps.append(cap_reference.get(cv2.CAP_PROP_POS_MSEC)*1e-3)
                elif len(timestamps) == 0:
                    overall_scores.append(overall_score*100)  
                    timestamps.append(cap_reference.get(cv2.CAP_PROP_POS_MSEC)*1e-3)
            score_color = color_gradient(overall_score)

            # Display the overall score on the top right corner of the live video
            cv2.putText(frame_live, f"Overall Score: {overall_score*100:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 5, cv2.LINE_AA)

            cv2.imshow("Reference Video", frame_reference)
            cv2.imshow("Live Video", frame_live)
            previous_ref_landmarks = reference_landmarks_smoothed
            previous_live_landmarks = landmarks_live_smoothed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting the loop.")
                break
    #Plot the overall scores through time
    plt.plot(timestamps,overall_scores)
    plt.xlabel("Timestamps(sec)")
    plt.ylabel("Score")
    ax=plt.gca()
    ax.set_ylim([0,100])
    plt.show()
    cap_reference.release()
    cap_live.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "source_video/uzun_video.mp4"
    compare_poses(video_path, use_live_camera=True)
