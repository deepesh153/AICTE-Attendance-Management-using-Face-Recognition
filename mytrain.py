import cv2  # importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

img = cv2.imread('couple2.jpg')#loading the image
cv2.imshow('Loaded Image', img)# showing the image
cv2.waitKey(0) #when wait key is 0, it stays infinitely open for long until it is manually closed
cv2.destroyAllWindows()

# grayscaling the image for identification
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.show()

#create a model using harcascade. without this file, face could not be identified
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#now detect the face(s). Keep scale factor between 0.8-1.4 max
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#here we will draw rectangle over face. the x, y are the left upper corner, x+w and y+h are the right bottom corner of the rectangle
for (x, y, w, h) in faces:
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h),color=(255, 0, 0),thickness= 2)

#now show the image with colour but using the indentified face coordinates to draw a square
plt.imshow(img_rgb)
plt.title('Detected Faces')
plt.show()

#from here we will train multiple images given in the folder
training_images = []  #create a list to hold training images
training_labels = []  #create a list to hold labels for each image

#now load all the images below using for loop. I have 5 images labelled from 1-5 so the range is 1-6
for i in range(1, 7):
    img = cv2.imread(f'train_image_{i}.jpg', cv2.IMREAD_GRAYSCALE)
    training_images.append(img)
    training_labels.append(i)  #once loaded the image, append the label name for the image in training label. when image will be identified, it will provide the same
    #image index for training labels also.

#check for the 1st array training image shape
training_images[0].shape
#check whether labels are properly appended.
training_labels

#now create LBPH recognizer to recognize the images and train using the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Train recognizer
recognizer.train(training_images, np.array(training_labels))

test_img = cv2.imread('test_face.jpg', cv2.IMREAD_GRAYSCALE)

#finally, predict the label of the test image. test image is obne of the image from the training images
label, confidence = recognizer.predict(test_img)
print(f'Label: {label}, Confidence: {confidence}')



#use videocapture to access the default camera using '0'. 0 means default camera
cap = cv2.VideoCapture(0)

while True:
    #here we will capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    #now take a look at the face in the opened window to display the live video stream. if you press 'c', it will capture your image and 'q' for quite which do not take photo
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        #and then save the captured image
        cv2.imwrite('captured_face.jpg', frame)
        print("Photo captured and saved as 'captured_face.jpg'")
        break
    elif key == ord('q'):
        break

#close windows and camera using release and destrpy
cap.release()
cv2.destroyAllWindows()

#just rename the captured image to a labeled format such that it do not have any random name. use below name to identify seperately from the known images
import os
os.rename('captured_face.jpg', 'person_1.jpg')
print("Photo renamed to 'person_1.jpg'")

#from here, same steps as performd earlier.
new_img = cv2.imread('person_1.jpg', cv2.IMREAD_GRAYSCALE)

#append to training data used earlier but using new label. because this image will need new label, not any label used earlier. here we are labeling it 7
training_images.append(new_img)
training_labels.append(7)  # Assign the same label as before

#now again train the recognizer with the updated data
recognizer.train(training_images, np.array(training_labels))
print("Model retrained with the newly captured image")

#now access the default camera again
cap = cv2.VideoCapture(0)

while True:
    #capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    #amd convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #now predict using the trained model (recognizer, used above to train)
    label, confidence = recognizer.predict(gray_frame)

    #display proper label and confidence on the frame
    cv2.putText(frame, f'Label: {label}, Confidence: {confidence}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #now show the live camera feed until you press 'q'
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()