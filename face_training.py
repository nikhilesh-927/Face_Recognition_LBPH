
import cv2
import numpy as np
from PIL import Image
import os

cascadePath = "haarcascade_frontalface_default.xml"

detector = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.LBPHFaceRecognizer.create() #LBPH pattern

def training(path):

    folderPaths = os.listdir(path)    
    faceSamples=[]
    ids = []
    count = 0
    for folder in folderPaths:
        folder_path = os.path.join(path,folder)
        folder_dirs = os.listdir(folder_path)
        if(folder_dirs): #check if folder dirs is not null
            for image in folder_dirs:
                imagePath = os.path.join(folder_path,image)
                image = cv2.imread(imagePath)
                image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                img_numpy = np.array(image_gray,'uint8')

                id = int(folder)
                faces = detector.detectMultiScale(img_numpy,scaleFactor = 1.3,minNeighbors = 5)
                if(len(faces)==1): #only allowed 1 face on an image
                    count+=1
                    for (x,y,w,h) in faces:
                        faceSamples.append(img_numpy[y:y+h,x:x+w])
                        ids.append(id)
    print(count)
    return faceSamples,ids
path = "dataset/"
faces,ids = training(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yaml
recognizer.write('trainer/trainer.yaml')

print("\n {0} faces trained. Thank you for waiting".format(len(np.unique(ids))))
