import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utility
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to access webcam.")
        break

    # it will Flip the image for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Show output
    cv2.imshow("MediaPipe Pose Detection", frame)

    # Press Q to quit rather than making deletion from terminal of task.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()