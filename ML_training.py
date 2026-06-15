import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier


df=pd.read_csv("heart.csv")

print(df)

def analyse():

    print(df.head(10))   

    print(df.tail(8))

    print(df.info())

    print(df.dtypes)

    print(df.describe())

    print(df.shape)
    print(df.columns)

    df.sample()

    print(df.corr())

    print(df['sex'].value_counts())

    df.hist(figsize=(12,10))

    plt.show()

    plt.figure(figsize=(10,8))

    sns.heatmap(df.corr(), annot=True)

    plt.show()

x=df.iloc[:,:13]
y=df['target']


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=47)


def svm(xtrain,xtest,ytrain,ytest):

    svm_clf=SVC()
    svm_clf.fit(xtrain,ytrain)
    y_predict=svm_clf.predict(xtest)

    print(y_predict)

    plt.scatter(
        xtrain.iloc[:,0],
        xtrain.iloc[:,1],
        c=ytrain,
        cmap='viridis'
    )

    plt.xlabel(xtrain.columns[1])
    plt.ylabel(xtrain.columns[2])
    plt.title("SVM Training Data Visualization")
    plt.show()

    acc=accuracy_score(ytest,y_predict)
    print(acc)
    cm=confusion_matrix(ytest,y_predict)
    print(cm)
    cr=classification_report(ytest,y_predict)
    print(cr)

def randomforest(xtrain,xtest,ytrain,ytest):

    raf=RandomForestClassifier()
    raf.fit(xtrain,ytrain)
    y_predict2=raf.predict(xtest)

    print(y_predict2)

    acc2=accuracy_score(ytest,y_predict2)
    print(acc2)
    cm2=confusion_matrix(ytest,y_predict2)
    print(cm2)
    cr2=classification_report(ytest,y_predict2)
    print(cr2)


def naiveBayas(xtrain,xtest,ytrain,ytest):

    raf=GaussianNB()
    raf.fit(xtrain,ytrain)
    y_predict3=raf.predict(xtest)

    print(y_predict3)

    acc3=accuracy_score(ytest,y_predict3)
    print(acc3)
    cm3=confusion_matrix(ytest,y_predict3)
    print(cm3)
    cr3=classification_report(ytest,y_predict3)
    print(cr3)

def decisiontree(xtrain,xtest,ytrain,ytest):

    raf=DecisionTreeClassifier()
    raf.fit(xtrain,ytrain)
    y_predict4=raf.predict(xtest)

    print(y_predict4)

    acc4=accuracy_score(ytest,y_predict4)
    print(acc4)
    cm4=confusion_matrix(ytest,y_predict4)
    print(cm4)
    cr4=classification_report(ytest,y_predict4)
    print(cr4)

def logistic(xtrain,xtest,ytrain,ytest):

    raf=LogisticRegression()
    raf.fit(xtrain,ytrain)
    y_predict5=raf.predict(xtest)

    print(y_predict5)

    acc5=accuracy_score(ytest,y_predict5)
    print(acc5)
    cm5=confusion_matrix(ytest,y_predict5)
    print(cm5)
    cr5=classification_report(ytest,y_predict5)
    print(cr5)

def Klearners(xtrain,xtest,ytrain,ytest):

    raf=KNeighborsClassifier()
    raf.fit(xtrain,ytrain)
    y_predict6=raf.predict(xtest)

    print(y_predict6)

    acc6=accuracy_score(ytest,y_predict6)
    print(acc6)
    cm6=confusion_matrix(ytest,y_predict6)
    print(cm6)
    cr6=classification_report(ytest,y_predict6)
    print(cr6)

while True:

    algorithm=input("enter 1 for svm or 2 for random forest or 3 for naiveBayas or 4 for decisiontree or 5 for logistic regression:")

    if algorithm=="1":
        
        analyse()

    elif algorithm=="2":

        svm(xtrain,xtest,ytrain,ytest)

    elif algorithm=="3":

        randomforest(xtrain,xtest,ytrain,ytest)

    elif algorithm=="4":

        naiveBayas(xtrain,xtest,ytrain,ytest)

    elif algorithm=="5":

        decisiontree(xtrain,xtest,ytrain,ytest)

    elif algorithm=="6":

        logistic(xtrain,xtest,ytrain,ytest)

    elif algorithm=="6":

        Klearners(xtrain,xtest,ytrain,ytest)

    else:

        print("Invalid option")

