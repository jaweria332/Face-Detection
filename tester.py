import cv2
import faceRecognition as fr


#This module takes images  stored in diskand performs face recognition
test_img=cv2.imread("E:\\TRY ON VIRTUAL SOFTWARE\\Step 1 Face Recognition\\test\\13.jpg")
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)


#uncomment the lines below when running the module first time as these will train the module
#faces,faceID=fr.labels_for_training_data('E:\\TRY ON VIRTUAL SOFTWARE\\Step 1 Face Recognition\\train')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.write('trainingData.yml')


#comment the lines when running the module first time
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#use this to load training data for subsequent runs
face_recognizer.read('trainingData.yml')

#creating dictionary containing names for each label
name={0:"Burak", 1:"Feroze"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    #predicting the label of given image
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    #Confidence shouldn't be more than 37, so if it is, module will skip it
    if(confidence>37):
        continue
    
    #calling function to put text on image as label
    fr.put_text(test_img,predicted_name,x,y)


#Resize the image after all processing
resized_img=cv2.resize(test_img,(750,750))
#Show the resultant image
cv2.imshow("Face Detection Module",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
