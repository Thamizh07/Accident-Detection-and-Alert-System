import cv2 as cv
import time
import geocoder
import os
from datetime import datetime
from pygame import mixer

mixer.init()
sound = mixer.Sound(r'C:\Users\thamizh\Desktop\sem 5\Six models\Pothole-Detection-System-using-YOLO-Tiny-v4-main\Pothole-Detection-System-using-YOLO-Tiny-v4-main\alarm.wav')

class_name = []
with open(r'C:\Users\thamizh\Desktop\sem 5\Six models\Pothole-Detection-System-using-YOLO-Tiny-v4-main\Pothole-Detection-System-using-YOLO-Tiny-v4-main\utils\obj.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net1 = cv.dnn.readNet(r'C:\Users\thamizh\Desktop\sem 5\Six models\Pothole-Detection-System-using-YOLO-Tiny-v4-main\Pothole-Detection-System-using-YOLO-Tiny-v4-main\utils\yolov4_tiny.weights', r'C:\Users\thamizh\Desktop\sem 5\Six models\Pothole-Detection-System-using-YOLO-Tiny-v4-main\Pothole-Detection-System-using-YOLO-Tiny-v4-main\utils\yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

cap = cv.VideoCapture(r"C:\Users\thamizh\Desktop\sem 5\Six models\Pothole-Detection-System-using-YOLO-Tiny-v4-main\Pothole-Detection-System-using-YOLO-Tiny-v4-main\test.mp4")
width  = cap.get(3)
height = cap.get(4)
result = cv.VideoWriter('result.avi',
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10, (int(width), int(height)))

g = geocoder.ip('me')
result_path = r"C:\Users\thamizh\Desktop\sem 5\Six models\Pothole-Detection-System-using-YOLO-Tiny-v4-main\Pothole-Detection-System-using-YOLO-Tiny-v4-main\pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break

    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)

    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        rec_area = w * h
        frame_area = width * height

        severity = "Low"
        if rec_area / frame_area > 0.1:
            severity = "High"
        elif rec_area / frame_area > 0.02:
            severity = "Medium"

        if len(scores) != 0 and scores[0] >= 0.7:
            if (rec_area / frame_area) <= 0.1 and box[1] < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.putText(frame, f"Severity: {severity}", (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

                if i == 0:
                    cv.imwrite(os.path.join(result_path, 'pot' + str(i) + '.jpg'), frame)
                    with open(os.path.join(result_path, 'pot' + str(i) + '.txt'), 'w') as f:
                        f.write(str(g.latlng) + f"\nSeverity: {severity}")
                    i += 1

                if i != 0:
                    if (time.time() - b) >= 2:
                        cv.imwrite(os.path.join(result_path, 'pot' + str(i) + '.jpg'), frame)
                        with open(os.path.join(result_path, 'pot' + str(i) + '.txt'), 'w') as f:
                            f.write(str(g.latlng) + f"\nSeverity: {severity}")
                        b = time.time()
                        i += 1

        if severity == 'High':
            sound.play()
        else:
            sound.stop()

    endingTime = time.time() - starting_time
    fps = frame_counter / endingTime
    cv.putText(frame, f'FPS: {fps}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv.imshow('frame', frame)
    result.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
result.release()
cv.destroyAllWindows()
