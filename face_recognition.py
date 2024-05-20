import cv2
import numpy as np
import os 
import time
import operator

recognizer = cv2.face.LBPHFaceRecognizer.create() #LBPH pattern
recognizer.read('trainer/trainer.yaml') #read the trained samples

cascadePath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['unknown']
start = time.time()
# Initialize and start realtime video capture
capture = cv2.VideoCapture(0)


# Define min window size to be recognized as a face
minW = 0.1*capture.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1*capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

start = time.time()

while True:
    #if(count == 120):
        #break


    ret, img =capture.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w]) #predict the user id and confidence

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 54):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):# press 'q' to quit
        break

capture.release()
cv2.destroyAllWindows()
