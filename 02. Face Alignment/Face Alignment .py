import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils
drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # Draw all facial landmarks
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # -------- Face Alignment --------

            h, w, _ = frame.shape

            nose = face_landmarks.landmark[1]
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]

            nose_x = nose.x * w
            left_x = left_cheek.x * w
            right_x = right_cheek.x * w

            face_center = (left_x + right_x) / 2

            difference = nose_x - face_center

            if difference < -15:
                direction = "Looking LEFT"

            elif difference > 15:
                direction = "Looking RIGHT"

            else:
                direction = "Looking CENTER"

            cv2.putText(
                frame,
                direction,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Alignment AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()