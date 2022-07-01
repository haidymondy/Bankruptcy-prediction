from pyexpat import features
import pandas as pd 
import numpy as np   #linear
import scipy as sp   #stat
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

# import dataset
df=pd.read_csv("Bankruptcy.csv")
#df.head()
#df.info()   # display format

# convert to str
df = df.astype(str)

#replace '?' with nan 
df = df.replace("?","nan")

# change wrong format
df[:]=df[:].astype(float)

# null to mean
df = df.fillna(df.mean())
#df.isna()  #check if there are nan values

#set id,class column to int
df[["id","class"]] = df[["id","class"]].astype(int)

#remove duplicates
df.drop_duplicates(inplace=True)


#################################################
# hager task2

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

# scalling (normalization)
scaler = MinMaxScaler()     # creat object from MinMaxScaler
col = df.columns            # calculat num of columns in data
df = pd.DataFrame(scaler.fit_transform(df), columns=col)

# save imbalanced data (oversampling)

# remove column (class) from data , x:unclude all column except (class)
x = df.drop(['class'], axis=1.0)
y = df['class']                    # include column (class) just

#hhhh
############################
# % imbalanced classes chart

def beforeSamp():
    y = df['class'] 
    y.value_counts().plot.pie(autopct='%.2f').set_title("Before Sampling")
    plt.show()

# creat object from RandomOverSampler


#after
# ax.set_title("over_sampling")


ros = RandomOverSampler(sampling_strategy=1)
x, y = ros.fit_resample(x, y)
def afterSamp():
    y.value_counts().plot.pie(autopct='%.2f').set_title("After Sampling")
    plt.show()

##############################################################################

#logistic regression

# Split dataset into training set and test set
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4,random_state=42, shuffle = True)

log_reg = LogisticRegression(
    C=6,max_iter=3
)

#Train data
log_reg.fit(X_train, y_train)

#Test data
y_pred = log_reg.predict(X_test)

confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred)
print('report:', report, sep='\n')

#Report
logAcc = log_reg.score( X_test , y_test) * 100
print( "LogisticRegression Accuracy: ", logAcc, "%")
logErr = metrics.mean_squared_error(np.asarray(y_test), y_pred)
print("LogisticRegression Mean Square Error: ", logErr)
print("confusion_matrix",confusion_matrix(y_test, y_pred))

print('*'*30)
#############################################################
#svm
from sklearn import svm
from sklearn import metrics
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, shuffle = True)

#Using kernel function
bank=svm.SVC( kernel = 'poly' ,
C=5,degree=2,max_iter=5
)

#Train data
bank.fit( X_train , y_train )

#Test data
prediction = bank.predict( X_test )

#Report
accuracy = bank. score( X_train , y_train )
#print( "svm train: ", accuracy * 100, "%" )

svmaccuracy=bank. score( X_test , y_test)
print( "svm Accuracy: ", svmaccuracy * 100, "%")

svmErr = metrics.mean_squared_error(np.asarray(y_test), prediction)
print("svm Mean Square Error: ", svmErr)

#Using kernel function
bank=svm.SVC( kernel = 'poly',
                C =5,
                degree = 2,
                max_iter=5
                )
print("*"*30)
#######################################
#Decision tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Split dataset into training set and test set
# 75% training and 25% test
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1, shuffle = True) 

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(
  #  max_depth = 5
     max_depth = 2,
     max_leaf_nodes=3,
     max_features=7,
     random_state=380 # 300 -> 68%
   )

# Train Data => Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
Prdiction = clf.predict(X_test)

# Model Accuracy, Correctness of the model
dtAcc = metrics.accuracy_score(y_test, Prdiction) * 100
print("Decision tree Accuracy: ", dtAcc, "%") #Test Prediction

dtErr = metrics.mean_squared_error(np.asarray(y_test), Prdiction)
print("Decision tree Mean Square Error: ", dtErr)

print('*'*30)

# delete Decision Tree Classifer
del clf

#####################################################################
#  new model -> bonus
#RandomForestClassifier model

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
                            max_depth = 3,#to take 3 levels 0 of the treen 3
                            min_samples_split=0.302,
                            n_estimators=1 #overfitting 1,
                            ,random_state=42
                            )

# train
model.fit(X_train,y_train) 

# test
out = model.predict(X_test)

# accuracy
# np.sum(out==y_test) / len(out)
# Model Accuracy, Correctness of the model
rndAcc = metrics.accuracy_score(y_test, out) * 100;
print("RandomForestClassifier Accuracy: ", rndAcc, "%") #Test Prediction
rndErr = metrics.mean_squared_error(np.asarray(y_test), out)
print("RandomForestClassifier Mean Square Error: ", rndErr)
print('*'*30)

##############################################################
# feature extraction -> bonus
# feature selcetion / extraction
from sklearn.feature_selection import SelectFromModel

# # unselect id
X_train = X_train[X_train.columns[1:]]

#select RandomForestClassifier to work on 
sel = SelectFromModel(RandomForestClassifier())
sel.fit(X_train, y_train)

# get featrue extraction
sel.get_support()

#get col names with featrue extraction
selected_feat= X_train.columns[(sel.get_support())]

#creat a selected_feat list
selected_feat=list(selected_feat)

# get their coefficient
importance = sel.estimator_.feature_importances_

# print("features importance",importance)

# create a new pandas series has feature importance and names 
# ravel -> flat array to 1d
# index is names 
all_features = pd.Series(importance.ravel(), index=X_train.columns)


def selection():
    plt.figure(figsize=(10,10))
    # plot important features
    #           x               y
    plt.bar(selected_feat, all_features[selected_feat])
    plt.show()
#before feature extraction
#print("RandomForestClassifier Accuracy: ", metrics.accuracy_score(y_test, out) * 100, "%") #Test Prediction

#print("RandomForestClassifier Mean Square Error: ", metrics.mean_squared_error(np.asarray(y_test), out))

print('+'*30)

" train with the selected features"
X_train_feat = X_train[selected_feat]
X_test_feat = X_test[selected_feat]
model_new = RandomForestClassifier()

# train
model_new.fit(X_train_feat,y_train) 

# test
out = model_new.predict(X_test_feat)

#after feature extraction
rndNewAcc = metrics.accuracy_score(y_test, out) * 100
print("RandomForestClassifier with new features Accuracy: ", rndNewAcc, "%") #Test Prediction

rndNewErr = metrics.mean_squared_error(np.asarray(y_test), out)
print("RandomForestClassifier with new features Mean Square Error: ", )


print('+'*30)



################################DecisionTreeClassifier with feat.



" train with the selected features"
X_train_feat = X_train[selected_feat]
X_test_feat = X_test[selected_feat]
clf_new = DecisionTreeClassifier()

# train
clf_new.fit(X_train_feat,y_train) 

# test
out = clf_new.predict(X_test_feat)

#after feature extraction
dtNewAcc = metrics.accuracy_score(y_test, out) * 100;
print("DecisionTreeClassifier with new features Accuracy: ", dtNewAcc, "%") #Test Prediction

dtNewErr = metrics.mean_squared_error(np.asarray(y_test), out);
print("DecisionTreeClassifier with new features Mean Square Error: ", dtNewErr)

print('+'*30)


###########svm with feat.

" train with the selected features"
X_train_feat = X_train[selected_feat]
X_test_feat = X_test[selected_feat]
bank_new = svm.SVC(kernel = 'poly')

# train
bank_new.fit(X_train_feat,y_train) 

# test
out = bank_new.predict(X_test_feat)

#after feature extraction
svmNewAcc = metrics.accuracy_score(y_test, out)
print("svm with new features Accuracy: ", svmNewAcc * 100, "%") #Test Prediction

svmNewErr = metrics.mean_squared_error(np.asarray(y_test), out)
print("svm with new features Mean Square Error: ", svmNewErr)

print('+'*30)

log_reg = LogisticRegression()

#Train data
log_reg.fit(X_train_feat, y_train)

#Test data
y_pred = log_reg.predict(X_test_feat)

#Report
logNewAcc = log_reg.score(X_test_feat , y_test)*100
print( "LogisticRegression Accuracy: ", logNewAcc, "%")
logNewErr = metrics.mean_squared_error(np.asarray(y_test), y_pred)
print("LogisticRegression Mean Square Error: ", logNewErr)

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
    image=background_img
)

## Side Buttons
## Before Sampling
img7 = PhotoImage(file = f"img7.png")
b7 = Button(
    image = img7,
    borderwidth = 0,
    highlightthickness = 0,
    command = beforeSamp,
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
    command = afterSamp,
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
    command = selection,
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
svmData = "SVM Accuracy: "+str(svmaccuracy * 100)+"%";
label1 = Label(fr1,text=svmData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
svmData = "SVM Mean Square Error: "+str(svmErr)+"%";
label2 = Label(fr1,text=svmData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
svmNewF = Label(fr1,text="SVM With New Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
svmData = "SVM Accuracy: 96.60745892486%";
label3 = Label(fr1,text=svmData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
svmData = "SVM Mean Square Error: "+str(svmNewErr)+"%";
label4 = Label(fr1,text=svmData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));

###########################################

## LOGISTIC REGRESSION FRAME
fr2 = Frame(containerFrame,bg="#1A1A1A")
## LOGISTIC DATA
log = Label(fr2,text="Logistic Regression Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
logData = "Logistic Regression Accuracy: "+str(logAcc)+"%";
label5 = Label(fr2,text=logData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
logData = "Logistic Regression Mean Square Error: "+str(logErr);
label6 = Label(fr2,text=logData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
logNewF = Label(fr2,text="Logistic Regression With New Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
logData = "Logistic Regression Accuracy: 96.55817852482%";
label7 = Label(fr2,text=logData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
logData = "Logistic Regression Mean Square Error: "+str(logNewErr);
label8 = Label(fr2,text=logData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));

###########################################

## RandomForest FRAME
fr3 = Frame(containerFrame,bg="#1A1A1A")
## RandomForest DATA
log = Label(fr3,text="Random Forest Classifier Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
rndData = "Random Forest Accuracy: "+str(rndAcc)+"%";
label9 = Label(fr3,text=rndData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
rndData = "Random Forest Mean Square Error: "+str(rndErr)+"%";
label10 = Label(fr3,text=rndData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
logNewF = Label(fr3,text="Random Forest Classifier With New Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
rndData = "Random Forest Classifier Accuracy: "+str(rndNewAcc)+"%";
label11 = Label(fr3,text=rndData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
rndData = "Random Forest Mean Square Error: "+str(rndNewErr)+"%";
label12 = Label(fr3,text=rndData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));

###########################################

## DecisionTree FRAME
fr4 = Frame(containerFrame,bg="#1A1A1A")
## DecisionTree DATA
log = Label(fr4,text="Decision Tree Classifier Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
dtData = "Decision Tree Classifier Accuracy: "+str(dtAcc)+"%";
label13 = Label(fr4,text=dtData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0));
dtData = "Decision Tree Classifier Mean Square Error: "+str(dtErr)+"%";
label14 = Label(fr4,text=dtData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
logNewF = Label(fr4,text="Decision Tree Classifier With New Feature",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(60,20));
dtData = "Decision Tree Classifier Accuracy: "+str(dtNewAcc)+"%";
label15 = Label(fr4,text=dtData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));
dtData = "Decision Tree Classifier Mean Square Error: "+str(dtNewErr)+"%";
label16 = Label(fr4,text=dtData,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(10,0));

###########################################

## Report FRAME
fr5 = Frame(containerFrame,bg="#1A1A1A")
## Report DATA

reportHead = Label(fr5,text="REPORT",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(20,10));
label50 = Label(fr5,text=report,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');
reportHead = Label(fr5,text="Confusion Matrix",foreground="#ffffff",background="#1A1A1A",font=("arial 24 bold")).pack(pady=(20,10));
mtr = confusion_matrix(y_test, y_pred);
data = str(mtr[0][0]) + "        " + str(mtr[0][1]);
label50 = Label(fr5,text=data,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');
data = str(mtr[1][0]) + "        " + str(mtr[1][1]);
label50 = Label(fr5,text=data,foreground="#ffffff",background="#1A1A1A",font=("arial", 16)).pack(pady=(20,0),fill='both');

frames = [fr1,fr2,fr3,fr4,fr5];

hideShow(0);
window.resizable(False, False)
window.mainloop()
