import sklearn
import os
import sklearn.ensemble
import sklearn.datasets
import sklearn.linear_model
import sklearn.tree
import pandas as pd
import numpy as np
import pylab as pl
import image
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.datasets
x=[]
x_test=[]
y_test=[]
y=[]
ori="C:\\Users\\Ritvik\\Documents\\datasets\\DevanagariHandwrittenCharacterDataset\\Train"
os.chdir(ori)
dirs=os.listdir()
for dir in dirs:
    for file in os.listdir(dir):
        f=os.fsdecode(file)
        img=Image.open(dir+"\\"+file)
        # img=img.convert('L')
        img=np.array(img)
        x.append(img)
        y.append(dir)
x = np.array(x).reshape(len(y),1024)
os.chdir(ori)
clf=sklearn.ensemble.RandomForestClassifier(random_state=0)
clf.fit(x,y)
ori="C:\\Users\\Ritvik\\Documents\\datasets\\DevanagariHandwrittenCharacterDataset\\Test"
os.chdir(ori)
dirs=os.listdir()
'''for dir in dirs:
    for file in os.listdir(dir):
        f=os.fsdecode(file)
        img=Image.open(dir+"\\"+file)
        # img=img.convert('L')
        img=np.array(img)
        x_test.append(img)
        y_test.append(dir)
x_test = np.array(x_test).reshape(len(y_test),1024)

print(clf.score(x_test,y_test))'''
img=Image.open("C:\\Users\\Ritvik\\Documents\\unt.jpg")
img=img.convert('L')
img=np.array(img)
print(clf.predict(img.reshape(1,1024)))
