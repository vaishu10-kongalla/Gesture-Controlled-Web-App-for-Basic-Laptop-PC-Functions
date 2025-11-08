import cv2
import mediapipe
import numpy as np
import pydirectinput as p1
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MediaPipe and screen dimensions
initHand = mediapipe.solutions.hands
mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mediapipe.solutions.drawing_utils

def handLandmarks(colorImg):
    landmarkList = []
    landmarkPositions = mainHand.process(colorImg)
    landmarkCheck = landmarkPositions.multi_hand_landmarks
    if landmarkCheck:
        for hand in landmarkCheck:
            draw.draw_landmarks(img, hand, initHand.HAND_CONNECTIONS)
            h, w, c = img.shape
            for index, landmark in enumerate(hand.landmark):
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)
                landmarkList.append([index, centerX, centerY])
    return landmarkList

def fingers(landmarks):
    if len(landmarks) < 21:
        return [0, 0, 0, 0, 0]
    fingerTips = []
    tipIds = [4, 8, 12, 16, 20]
    if landmarks[tipIds[0]][1] > landmarks[tipIds[0] - 1][1]:
        fingerTips.append(1)
    else:
        fingerTips.append(0)
    for id in range(1, 5):
        if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 3][2]:
            fingerTips.append(1)
        else:
            fingerTips.append(0)
    return fingerTips

while True:
    check, img = cap.read()
    if not check:
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lmList = handLandmarks(imgRGB)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[4][1:]  # Thumb tip
        x3, y3 = lmList[12][1:]  # Middle finger tip

        finger = fingers(lmList)

        # W (Forward) - Index finger up
        if finger[1] == 1 and finger[2] == 0 and finger[3] == 0 and finger[4] == 0:
            p1.keyDown("w")
        else:
            p1.keyUp("w")

        # A (Left) - Thumb up
        if finger[0] == 1 and finger[1] == 0 and finger[2] == 0 and finger[3] == 0 and finger[4] == 0:
            p1.keyDown("a")
        else:
            p1.keyUp("a")

        # S (Backward) - Middle finger up
        if finger[2] == 1 and finger[1] == 1 and finger[3] == 0 and finger[4] == 0:
            p1.keyDown("s")
        else:
            p1.keyUp("s")

        # D (Right) - Ring finger up
        if finger[4] == 1 and finger[1] == 0 and finger[2] == 0 and finger[3] == 0:
            p1.keyDown("d")
        else:
            p1.keyUp("d")

        # Left Mouse Click - Thumb and Index finger together
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance < 30:
            p1.click(button='left')

        # Right Mouse Click - Thumb and Middle finger together
        distance_middle = math.hypot(x2 - x3, y2 - y3)
        if distance_middle < 30:
            p1.click(button='right')

    # Display instructions
    cv2.putText(img, "W: Index Up | A: Thumb Up | S: Middle Up | D: Ring Up", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.putText(img, "Left Click: Thumb + Index | Right Click: Thumb + Middle", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.putText(img, "Press 'Q' to quit", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    # Show the image
    cv2.imshow("Hand Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()