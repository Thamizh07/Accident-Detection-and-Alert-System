import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import warnings

warnings.filterwarnings("ignore")

mixer.init()
sound = mixer.Sound(r'C:\Users\thamizh\Desktop\sem 5\Six models\Drowsiness detection\alarm.wav')

face_cascade_path = r'C:\Users\thamizh\Desktop\sem 5\Six models\Drowsiness detection\haar cascade files\haarcascade_frontalface_alt.xml'
leye_cascade_path = r'C:\Users\thamizh\Desktop\sem 5\Six models\Drowsiness detection\haar cascade files\haarcascade_lefteye_2splits.xml'
reye_cascade_path = r'C:\Users\thamizh\Desktop\sem 5\Six models\Drowsiness detection\haar cascade files\haarcascade_righteye_2splits.xml'

model_path = r'C:\Users\thamizh\Desktop\sem 5\Six models\Drowsiness detection\models\cnnCat2.h5'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
leye_cascade = cv2.CascadeClassifier(leye_cascade_path)
reye_cascade = cv2.CascadeClassifier(reye_cascade_path)

lbl=['Close','Open']

model = load_model(model_path)
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[[2,2]]
lpred=[[2,2]]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        left_eye = leye_cascade.detectMultiScale(roi_gray)
        right_eye = reye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in right_eye:
            r_eye=roi_color[ey:ey+eh, ex:ex+ew]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = (model.predict(r_eye) > 0.5).astype("int32").tolist()
            if(rpred[0][0]<rpred[0][1]):
                lbl='Open' 
            else:
                lbl='Closed'
            break

        for (ex,ey,ew,eh) in left_eye:
            l_eye=roi_color[ey:ey+eh, ex:ex+ew]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = (model.predict(l_eye) > 0.5).astype("int32").tolist()
            if(lpred[0][0]<lpred[0][1]):
                lbl='Open'   
            else:
                lbl='Closed'
            break
    
    if(rpred[0][1]<rpred[0][0]) and (lpred[0][1]<lpred[0][0]):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>35):
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()    
        except:  
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    else:
        sound.stop()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
