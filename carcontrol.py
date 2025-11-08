import cv2
import mediapipe
import numpy as np
import autopy
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
wScr, hScr = autopy.screen.size()
pX, pY = 0, 0
cX, cY = 0, 0

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
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        finger = fingers(lmList)

        # Mouse movement
        if finger[1] == 1 and finger[2] == 0 and finger[4] == 0:
            hCam, wCam, _ = img.shape
            x3 = np.interp(x1, (75, wCam - 75), (0, wScr))
            y3 = np.interp(y1, (75, hCam - 75), (0, hScr))
            cX = pX + (x3 - pX) / 7
            cY = pY + (y3 - pY) / 7
            autopy.mouse.move(wScr - cX, cY)
            pX, pY = cX, cY

        # Left mouse click
        if finger[1] == 0 and finger[0] == 1:
            p1.click(button='left')

        # Right key press
        if sum(finger) == 5:
            p1.keyDown("right")
            p1.keyUp("left")

        # Left key press
        elif sum(finger) == 0:
            p1.keyDown("left")
            p1.keyUp("right")

        # Space key press
        elif finger[1] == 1 and finger[2] == 1 and finger[3] == 1:
            p1.press("space")

        # Forward command (index and middle fingers up, others down)
        elif finger[1] == 1 and finger[2] == 1 and finger[3] == 0 and finger[4] == 0:
            p1.keyDown("up")  # Simulate forward movement
            p1.keyUp("down")

        # Reset keys
        elif finger[1] == 1:
            p1.keyUp("right")
            p1.keyUp("left")
            p1.keyUp("up")
            p1.keyUp("down")

    # Display instructions
    cv2.putText(img, "Press 'Q' to quit", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()