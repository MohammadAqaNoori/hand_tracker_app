import cv2
import mediapipe as mp

# Initialize Mediapipe and drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Setup camera
cap = cv2.VideoCapture(0)

# Initialize hand detection
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(frame_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                landmarks = handLms.landmark

                # Finger tip landmark IDs (for fingers: [thumb, index, middle, ring, pinky])
                tip_ids = [4, 8, 12, 16, 20]

                fingers = []

                # Thumb
                if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other four fingers
                for id in range(1, 5):
                    if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)

                # Show finger count and gesture
                gesture = ""
                if total_fingers == 0:
                    gesture = "Fist"
                elif total_fingers == 5:
                    gesture = "Open Palm"
                elif total_fingers == 2 and fingers[1] and fingers[2]:
                    gesture = "Peace ✌️"
                else:
                    gesture = f"{total_fingers} Fingers Up"

                cv2.putText(frame, gesture, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow('Hand Tracker', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
