import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Hands 模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化 MediaPipe Drawing 模組
mp_drawing = mp.solutions.drawing_utils

# 開啟攝影機
cap = cv2.VideoCapture(0)

def apply_mosaic(frame, x, y, width, height, block_size=10):
    """ 在指定區域內應用馬賽克效果 """
    # 確保區域不會超出影像邊界
    x, y = max(0, x), max(0, y)
    width, height = min(frame.shape[1] - x, width), min(frame.shape[0] - y, height)

    # 檢查區域大小是否為正數
    if width > 0 and height > 0:
        # 取得區域
        roi = frame[y:y+height, x:x+width]
        # 將區域縮小並再放大回去，形成馬賽克
        small = cv2.resize(roi, (width // block_size, height // block_size), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        # 替換回原區域
        frame[y:y+height, x:x+width] = mosaic
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 取得中指的三個關鍵點
            middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]  # 中指根部
            middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]  # 中指中段
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]  # 中指尖端

            # 取得三個關鍵點的位置
            points = [
                (int(middle_finger_mcp.x * w), int(middle_finger_mcp.y * h)),
                (int(middle_finger_pip.x * w), int(middle_finger_pip.y * h)),
                (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h))
            ]

            # 繪製三個中指關鍵點
            for point in points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)  # 綠色點標示中指的關鍵點

            # 針對每個關鍵點周圍區域應用馬賽克效果
            block_size = 20  # 調整馬賽克區塊大小
            for point in points:
                x, y = point
                apply_mosaic(frame, x - block_size, y - block_size, block_size * 2, block_size * 2)

            # 繪製手部骨架連線
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 顯示影像
    cv2.imshow("Hand Joint Detection with Mosaic on Middle Finger", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
