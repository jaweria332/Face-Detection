#Importing necessary libraries
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image
import face_recognition
import os
from datetime import datetime

root = Tk()

root.title("Live Recognition system | Develop by Jaweria")
root.geometry("1350x700+0+0")
root.config(bg="black")

path = "E:\\TRY_ON_VIRTUAL\\Recognition_System\\dataset"
images = []
classNames = []
myList = os.listdir(path)

#Defining functions
def generate_dataset():
    if (e1.get() == "" or e2.get() == ""):
        messagebox.showerror('Error', 'Please provide complete details of the user')
    else:
        haar_file = "E:\\TRY_ON_VIRTUAL\\Step_1_face_recognize\\haarcascade_frontalface_default.xml"
        dataset = 'dataset'
        # Which person dataset
        subdata = str(l1_var.get())

        path = os.path.join(dataset, subdata)
        # If path is available or not
        if not os.path.isdir(path):
            os.mkdir(path)

        (width, height) = (130, 100)
        # uploaded haar alogrithm to classifier
        face_cascade = cv2.CascadeClassifier(haar_file)
        # initialize camera
        webcam = cv2.VideoCapture(0)
        print(webcam)
        count = 1
        while count < 51:
            print(count)
            (ret, img) = webcam.read()

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale img
            facesFound = face_cascade.detectMultiScale(gray_img, 1.32, 3)  # get coordinates of faces
            for (x, y, w, h) in facesFound:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle arround faces found
                faceFound = gray_img[y:y + h, x:x + w]  # crop face part for dataset
                face_resized = cv2.resize(faceFound, (width, height))
                cv2.imwrite('%s/%s.png' % (path, count), face_resized)
            count += 1

            cv2.imshow('Face', img)
            key = cv2.waitKey(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!!!')


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
    return dtString

def train_classifier():
    haar_file = "E:\\TRY_ON_VIRTUAL\\Step_1_face_recognize\\haarcascade_frontalface_default.xml"
    dataset = 'dataset'
    (images, labels, names, id) = ([], [], {}, 0)

    # {dir:datasetfolder, subdirs:[Person1,Person2,Person3],files:images of each person

    for (subdirs, dirs, files) in os.walk(dataset):
        for subdir in dirs:  # Loop through each person folder
            names[id] = subdir  # Assining name of person , name[0]=person1
            subjectpath = os.path.join(dataset, subdir)  # dataset/person1
            for filename in os.listdir(subjectpath):  # Loop through each img name
                imgpath = subjectpath + '/' + filename  # dataset/person1/1.png
                label = id
                # Adding images from dataset in imgage var in grayscale
                images.append(cv2.imread(imgpath, 0))
                # Adding respective label
                labels.append(int(label))

            id += 1

            # images = contain now 60 images and we have 60 labels
    (width, height) = (130, 100)
    (images, labels) = [np.array(lis) for lis in [images, labels]]
    print(images, labels)
    # Load the Model
    face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
    # train the model
    face_recognizer_model.train(images, labels)

    messagebox.showinfo('Result', 'Training dataset completed!!!')
    return classNames, images


def start_detection():
    def draw_rect(img, facesFound):
        (x, y, w, h) = facesFound
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)

    def put_text(img, text, x, y):
        cv2.putText(img, text, (x - 10, y + 230), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    haar_file = "E:\\TRY_ON_VIRTUAL\\Step_1_face_recognize\\haarcascade_frontalface_default.xml"
    dataset = 'dataset'
    (images, labels, names, id) = ([], [], {}, 0)

    # {dir:datasetfolder, subdirs:[Person1,Person2,Person3],files:images of each person

    for (subdirs, dirs, files) in os.walk(dataset):
        for subdir in dirs:  # Loop through each person folder
            names[id] = subdir  # Assining name of person , name[0]=person1
            subjectpath = os.path.join(dataset, subdir)  # dataset/person1
            for filename in os.listdir(subjectpath):  # Loop through each img name
                imgpath = subjectpath + '/' + filename  # dataset/person1/1.png
                label = id
                # Adding images from dataset in imgage var in grayscale
                images.append(cv2.imread(imgpath, 0))
                # Adding respective label
                labels.append(int(label))

            id += 1

            # images = contain now 60 images and we have 60 labels
    (width, height) = (130, 100)
    (images, labels) = [np.array(lis) for lis in [images, labels]]
    print(images, labels)
    # Load the Model
    face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
    # train the model
    face_recognizer_model.train(images, labels)
    messagebox.showinfo("Information","Module is ready to detect")
    face_cascade = cv2.CascadeClassifier(haar_file)

    # initialize camera
    webcam = cv2.VideoCapture(0)
    cnt = 0
    while True:  # Capturing 30 images
        (ret, img) = webcam.read()  # read camera
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale img
        facesFound = face_cascade.detectMultiScale(gray_img, 1.32, 3)  # get coordinates of faces
        for (x, y, w, h) in facesFound:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle arround faces found
            faceFound = gray_img[y:y + h, x:x + w]  # crop face part for dataset
            face_resized = cv2.resize(faceFound, (width, height))
            label, confidence = face_recognizer_model.predict(face_resized)
            draw_rect(img, (x, y, w, h))
            predicted_name = names[label]

            print(predicted_name)
            print(confidence)

            put_text(img, predicted_name, x, y)
        cv2.imshow('Opencv', img)
        key = cv2.waitKey(10)
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


# from PIL import ImageGrab
lbl_title = Label(root, text="RECOGNITION SYSTEM", font=("times new roman", 40, "bold"))
lbl_title.place(x=0, y=0, relwidth=1)

# Defining a frame
reg_frame = Frame(root, bg="#262626")
reg_frame.place(x=20, y=100, width=600, height=600)

det_frame = Frame(root, bg="#262626")
det_frame.place(x=630, y=100, width=600, height=600)

#Defining label and entries for input
l1=Label(reg_frame, text="Enter name : ", font=("times new roman", 16, "bold"), bg="#262626", fg="white")
l1.place(x=10, y=10)
l1_var=StringVar()
e1=Entry(reg_frame, textvariable=l1_var, font=("times new roman", 16, "bold"), bg="white", fg="black")
e1.place(x=150, y=10)

l2=Label(reg_frame, text="Enter Gender : ", font=("times new roman", 16, "bold"), bg="#262626", fg="white")
l2.place(x=10, y=60)
e2=Entry(reg_frame, font=("times new roman", 16, "bold"), bg="white", fg="black")
e2.place(x=150, y=60)

#Placing button for generating dataset
gnrt_btn=Button(reg_frame, text="Generate Dataset", command=generate_dataset, font=("times new roman" , 20, "normal"), bg="orange", fg="brown", activebackground="lightgreen", activeforeground="green")
gnrt_btn.place(x=200, y=120, width=200, height=40)

lbl_txt=Label(reg_frame, text="You can start training after generating dataset", font=("courier new" , 14, "normal"), bg="#262626", fg="yellow")
lbl_txt.place(x=20, y=180)

#Placing button for training
gnrt_btn=Button(reg_frame, text="start training",command=train_classifier, font=("times new roman" , 20, "normal"), bg="pink", fg="red", activebackground="lightgreen", activeforeground="green")
gnrt_btn.place(x=200, y=250, width=200, height=40)

#Place button to start recognizing
recog_btn=Button(det_frame, text="Start Detection",command=start_detection, font=("times new roman" , 20, "normal"), bg="orange", fg="red", activebackground="lightgreen", activeforeground="green")
recog_btn.place(x=200, y=50, width=200, height=40)

# Place label to stop recognizing
recog_btn = Label(det_frame, text="Press q to exit detection",
                   font=("courier new", 14, "normal"), bg="#262626", fg="yellow", bd=None)
recog_btn.place(x=80, y=10, width=400)

root.mainloop()