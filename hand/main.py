import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0) #鏡頭編號
mpHands = mp.solutions.hands #使用手部模型
hands = mpHands.Hands() #呼叫函式
draw_mark = mp.solutions.drawing_utils #畫點

while True:
    boo, nex = cam.read() #boo為是否取得成功 nex為下一偵的圖片
    if boo:
        nex_rgb = cv2.cvtColor(nex, cv2.COLOR_BGR2RGB) #預設為BGR 模型使用RGB
        result = hands.process(nex_rgb) #將結果存在result
        
        nex_x = nex.shape[1] #視窗長寬
        nex_y = nex.shape[0]
        
        if result.multi_hand_landmarks: #若畫面有手
            for handLms in result.multi_hand_landmarks:
                draw_mark.draw_landmarks(nex, handLms, mpHands.HAND_CONNECTIONS) #(每一偵，畫點，畫線)
                for i, lm in enumerate(handLms.landmark): #遍歷每次偵測手的點
                    x_pos = int(lm.x * nex_x) #根據長寬換算整數座標
                    y_pos = int(lm.y * nex_y)
                    cv2.putText(nex, str(i), (x_pos-25, y_pos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2) #再點上標記編號(圖,編號,位置,字形,大小,顏色,粗度)
                    print(i, x_pos, y_pos) #印出座標

        cv2.imshow("video",nex)

    else:
        break

    if cv2.waitKey(1) == 27: #ESC結束
        break