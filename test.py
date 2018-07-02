#packages
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
#import tkinter as tk
from tkinter import *
from functools import partial

def read_data():
	odf=pd.read_csv('temp.csv')
	return odf

def get_headers(dataset):
	return (dataset.columns.values)
	
def split_dataset(dataset,train_percentage,feature_headers,target_headers):
	train_x,test_x,train_y,test_y = train_test_split(dataset[feature_headers],dataset[target_headers],train_size=train_percentage)
	return train_x,test_x,train_y,test_y


def random_forest_classifier(features,targets):
	clf=RandomForestClassifier()
	clf.fit(features,targets)
	return clf

def Naive_bayes(features,targets):
	clf=GaussianNB()
	clf.fit(features,targets)
	return clf

def linear_regression(features,targets):
	clf=linear_model.LinearRegression()
	clf.fit(features,targets)
	return clf
    
def final(arg,features,targets):
	switcher = {
	1:Naive_bayes(features,targets),
	2:random_forest_classifier(features,targets),
	3:linear_regression(features,targets),
	}
	return switcher.get(arg,"Nothing")


global call_result

def main():
	dataset=read_data()
	HEADERS=get_headers(dataset)

	train_x,test_x,train_y,test_y=split_dataset(dataset,0.99,HEADERS[1:-1],HEADERS[-1])
	print("train_x shape: ",train_x.shape)
	print("train_y shape: ",train_y.shape)
	print("test_x shape: ",test_x.shape)
	print("test_y shape: ",test_y.shape)
	def call_result(label_result,option,features,targets,test_x):
		option=(option.get())
		clf=final(option,features,targets)
		print(clf)
		result=clf.predict(test_x)
		str1='\n'.join(str(x) for x in result)
		label_result.config(text="%s"%str1)
		return 

	#tkinter initialization
	root =Tk()
	root.geometry('400x400')
	root.title("TEST")

	option = IntVar()
	labelResult =Label(root)
	labelResult.grid(row=8, column=2)

	# radio buttons
	R1=Radiobutton(root,text="Naive_bayes",variable=option,value=1).grid(row=1,column=1)
	R2=Radiobutton(root,text="Random Forest Classifier",variable=option,value=2).grid(row=2,column=1)
	R3=Radiobutton(root,text="Linear Regression",variable=option,value=3).grid(row=3,column=1)

	call_result = partial(call_result,labelResult,option,train_x,train_y,test_x)
	buttonCal=Button(root,text="Find",command=call_result).grid(row=4,column=0)

	root.mainloop()
main()


	
