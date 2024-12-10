import cv2
import os
import numpy as np
from PIL import Image

#recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImageAndLabels(path):
 #get the path of all the files in the folder
 imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
 #create empty list for faces
 faceSamples = []
 #create empty ID list
 Ids = []
 #now looping through all the image paths and loading the ids and the images 
 for imagePath in imagePaths:
  #loading image and converting it into grayScale
  pilImage = Image.open(imagePath).convert('L')
  #converting the PIL image to numpy array
  imageNp = np.array(pilImage,'uint8')

  #getting the Id from the image
  Id = int(os.path.split(imagePath)[-1].split(".")[1])
  #extracting the face from training image sample
  faces = detector.detectMultiScale(imageNp)
  #if the face exists then append it with its id
  for(x,y,w,h) in faces:
   faceSamples.append(imageNp[y:y+h,x:x+w])
 return faceSamples, Ids

faces, Ids = getImageAndLabels('TrainingImage')
recognizer.train(faces,np.array(Ids))
recognizer.save(('TrainingImageLabel/trainner.yml'))
