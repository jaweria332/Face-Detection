import cv2
import os
import numpy as np

#This module contains all common functions used in tester.py file


#read image, convert it into grayscale and return it
def faceDetection(test_img):
    #convert image into grayscale
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    #Import haarcascade classifier
    face_haar_cascade=cv2.CascadeClassifier('E:\\TRY ON VIRTUAL SOFTWARE\\Step 1 Face Recognition\\haarcascade_frontalface_default.xml')
    #Detect multiscale
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)
    #return resultant image
    return faces,gray_img

#Given a directory below function returns part of gray_img which is face alongwith its label/ID
def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                #Skipping files that startwith .
                print("Skipping system file whose name start with . ") 
                continue
            
            #fetching subdirectory names
            id=os.path.basename(path)
            #fetching image path
            img_path=os.path.join(path,filename)
            #print image path and ID
            print("Image path : ",img_path)
            print("ID : ",id)
            #load image serially
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image is not loaded properly")
                continue
                
                
            #Calling faceDetection function to return faces detected in particular image
            faces_rect,gray_img=faceDetection(test_img)
            
            #if more than one faces are detected in any image skip it
            if len(faces_rect)!=1:
               continue
            (x,y,w,h)=faces_rect[0]
            
            #cropping region of interest i.e. face area from grayscale image
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

#Below function draws bounding boxes around detected face in image
def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

#Below function writes name of person for detected label
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
