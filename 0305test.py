import cv2
import mediapipe as mp

# 初始化 Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 設定特定部位的索引
RIGHT_EYE = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
LEFT_EYE = [263, 373, 380, 362, 385, 387, 33, 160, 158, 133, 153, 144]
NOSE = [1, 2, 98, 327, 195, 5, 4, 275, 440]
MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146,78, 191, 80, 81, 82, 13, 311, 308, 415, 310, 317, 14]
# 讀取攝影機畫面
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換 BGR 影像為 RGB（Mediapipe 使用 RGB）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 進行人臉偵測
    results = face_mesh.process(rgb_frame)

    # 如果偵測到人臉
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # 畫眼睛（藍色）
            for idx in RIGHT_EYE + LEFT_EYE:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # 藍色點

            # 畫鼻子（紅色）
            for idx in NOSE:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # 紅色點

            # 畫嘴巴（黃色，外圍）
            for idx in MOUTH:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # 黃色點（外圍）

    # 顯示畫面
    cv2.imshow("Face Parts Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
