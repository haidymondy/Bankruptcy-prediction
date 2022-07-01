import tkinter as ttk
from tkinter import *
from tkinter import font
from turtle import width

from matplotlib.pyplot import margins

def hideShow(x):
    for i in frames:
        i.pack_forget()
    frames[x].pack()



window = Tk();

window.geometry("1243x645")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 645,
    width = 1243,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    613.5, 327.5,
    image=background_img)

## Side Buttons
## Before Sampling
img7 = PhotoImage(file = f"img7.png")
b7 = Button(
    image = img7,
    borderwidth = 0,
    highlightthickness = 0,
    relief = "flat")
b7.place(
    x = 0, y = 67,
    width = 256,
    height = 64)

## After Sampling
img6 = PhotoImage(file = f"img6.png")
b6 = Button(
    image = img6,
    borderwidth = 0,
    highlightthickness = 0,
    relief = "flat")
b6.place(
    x = 0, y = 134,
    width = 256,
    height = 64)

## Feature Sellection
img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command=lambda x=0 : hideShow(x),
    relief = "flat")
b0.place(
    x = 0, y = 200,
    width = 256,
    height = 64)

## Random Forest
img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda x=2 : hideShow(x),
    relief = "flat")
b1.place(
    x = 0, y = 516,
    width = 256,
    height = 64)

## Report
img2 = PhotoImage(file = f"img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda x=4 : hideShow(x),
    relief = "flat")
b2.place(
    x = 0, y = 581,
    width = 256,
    height = 64)

## Decision Tree
img3 = PhotoImage(file = f"img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda x=3 : hideShow(x),
    relief = "flat")
b3.place(
    x = 0, y = 451,
    width = 257,
    height = 64)

## LOGISTIC
img4 = PhotoImage(file = f"img4.png")
b4 = Button(
    image = img4,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda x=1 : hideShow(x),
    relief = "flat")
b4.place(
    x = 0, y = 386,
    width = 256,
    height = 64)

## SVM
img5 = PhotoImage(file = f"img5.png")
b5 = Button(
    image = img5,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda x=0 : hideShow(x),
    relief = "flat")
b5.place(
    x = 0, y = 321,
    width = 256,
    height = 64)

## Main Viewer
containerFrame = Frame(window,bg="#1A1A1A")
containerFrame.place(
    x=265,
    y=80,
    width=970,
    height=550);

## SVM FRAME
fr1 = Frame(containerFrame,bg="#1A1A1A")
## SVM DATA
svm = Label(fr1,text="SVM Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
label1 = Label(fr1,text="SVM Accuracy:  97.85138764547897%",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
label2 = Label(fr1,text="SVM Mean Square Error:  0.021486123545210387",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
svmNewF = Label(fr1,text="SVM With New Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
label3 = Label(fr1,text="SVM Accuracy:  50.37728609796649 %",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
label4 = Label(fr1,text="SVM Mean Square Error:  0.4962271390203351",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));

###########################################

## LOGISTIC REGRESSION FRAME
fr2 = Frame(containerFrame,bg="#1A1A1A")
## LOGISTIC DATA
log = Label(fr2,text="Logistic Regression Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
label5 = Label(fr2,text="Logistic Regression Accuracy:  96.72592403120603%",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
label6 = Label(fr2,text="Logistic Regression Mean Square Error:  0.03274075968793964",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
logNewF = Label(fr2,text="Logistic Regression With New Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
label7 = Label(fr2,text="Logistic Regression Accuracy:  49.72502877605832%",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
label8 = Label(fr2,text="Logistic Regression Mean Square Error:  0.5027497122394168",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));

###########################################

## RandomForest FRAME
fr3 = Frame(containerFrame,bg="#1A1A1A")
## RandomForest DATA
log = Label(fr3,text="Random Forest Classifier Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
label9 = Label(fr3,text="Random Forest Classifier Accuracy:  96.70034531269984%",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
label10 = Label(fr3,text="Random Forest Classifier Mean Square Error:  0.032996546873001666",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
logNewF = Label(fr3,text="Random Forest Classifier With New Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
label11 = Label(fr3,text="Random Forest Classifier Accuracy:  99.83373832970969%",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
label12 = Label(fr3,text="Random Forest Classifier Mean Square Error:  0.0016626167029031844",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));

###########################################

## DecisionTree FRAME
fr4 = Frame(containerFrame,bg="#1A1A1A")
## DecisionTree DATA
log = Label(fr4,text="Decision Tree Classifier Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
label13 = Label(fr4,text="Decision Tree Classifier Accuracy:  84.7934518480624%",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
label14 = Label(fr4,text="Decision Tree Classifier Mean Square Error:  0.15244916229696892",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
logNewF = Label(fr4,text="Decision Tree Classifier With New Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
label15 = Label(fr4,text="Decision Tree Classifier Accuracy:  98.1327535490472%",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
label16 = Label(fr4,text="Decision Tree Classifier Mean Square Error:  0.018672464509528072",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));

###########################################

## Report FRAME
fr5 = Frame(containerFrame,bg="#1A1A1A")
## Report DATA
reportHead = Label(fr5,text="REPORT",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(80,20));
label50 = Label(fr5,text="                     precision    recall   f1-score   support",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');
label51 = Label(fr5,text="0.0                   1.00       0.93         0.97        3930 ",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');
label51 = Label(fr5,text="1.0                   0.94       1.00         0.97        3889 ",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');
label51 = Label(fr5,text="Accuracy                                        0.97        7819 ",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');
label51 = Label(fr5,text="Macro avg       0.97       0.97         0.97        7819",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');
label51 = Label(fr5,text="Weighted avg    0.97       0.97         0.97        7819 ",foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');

frames = [fr1,fr2,fr3,fr4,fr5];

hideShow(0);
window.resizable(False, False)
window.mainloop()
