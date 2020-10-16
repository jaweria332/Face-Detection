import os
import numpy
import cv2



def draw_rect(img, facesFound):
    (x, y, w, h) = facesFound
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)


def put_text(img, text, x, y):
    cv2.putText(img, text, (x -10, y + 230), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0),2 )


haar_file="E:\\TRY_ON_VIRTUAL\\Step_1_face_recognize\\haarcascade_frontalface_default.xml"
dataset='datasets'
(images,labels,names,id)=([],[],{},0)

# {dir:datasetfolder, subdirs:[Person1,Person2,Person3],files:images of each person

for (subdirs,dirs,files) in os.walk(dataset):
    for subdir in dirs: #Loop through each person folder
        names[id]=subdir #Assining name of person , name[0]=person1
        subjectpath=os.path.join(dataset,subdir)  #dataset/person1
        for filename in os.listdir(subjectpath): #Loop through each img name
            imgpath=subjectpath+'/'+filename #dataset/person1/1.png
            label=id
            #Adding images from dataset in imgage var in grayscale
            images.append(cv2.imread(imgpath,0))
            #Adding respective label
            labels.append(int(label))

        id+=1  

#images = contain now 60 images and we have 60 labels
(width,height)=(130,100)
(images,labels)=[numpy.array(lis) for lis in [images,labels]]
print(images,labels)
# Load the Model
face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
#train the model
face_recognizer_model.train(images,labels)
print('Training Complete')

face_cascade=cv2.CascadeClassifier(haar_file)


#initialize camera
webcam=cv2.VideoCapture(0)
cnt=0
while True: #Capturing 30 images
    (ret,img)=webcam.read() #read camera
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)              # grayscale img
    facesFound= face_cascade.detectMultiScale(gray_img,1.32,3) #get coordinates of faces
    for (x,y,w,h) in facesFound:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)         #draw rectangle arround faces found
        faceFound=gray_img[y:y+h,x:x+w] #crop face part for dataset
        face_resized=cv2.resize(faceFound,(width,height))
        label,confidence= face_recognizer_model.predict(face_resized)
        draw_rect(img, (x,y,w,h))
        predicted_name = names[label]

        print(predicted_name)
        print(confidence)

        put_text(img, predicted_name, x, y)
    cv2.imshow('Opencv',img)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()




