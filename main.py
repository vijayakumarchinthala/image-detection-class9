# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import tkinter 
from tkinter import filedialog
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import  preprocess_input,decode_predictions
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

model = DenseNet121(weights='imagenet')

app = tkinter.Tk()
app.geometry("750x625")
app.title(" Class 9 Image Classifier")

def getImage():
    path = filedialog.askopenfilename()
    return path

def preprocess(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expanded = numpy.expand_dims(img_array, axis=0)
    img_processed = preprocess_input(img_expanded)
    return img_processed

def classify(processed_img):
    preds = model.predict(processed_img)
    results = decode_predictions(preds, top=5)[0]
    return results

def image_prediction():
    path = getImage()
    input_img = preprocess(path)
    results = classify(input_img)
    result_label.config(text="")
    result_label.config(text=results[0][1] + ": " + str(results[0][2]))


# Create an object of tkinter ImageTk
image1= ImageTk.PhotoImage(Image.open(r"C:\Users\rainb\OneDrive\Documents\Desktop\main.jpg"))
# Create a Label Widget to display the text or Image
Label(app, image = image1).pack()

#Create Label
Label(app, text="Class 9 AI objection detection demo", font=(
	"Helvetica 10 bold"), fg="blue",bg="orange").pack()

upload_button = tkinter.Button(text="Upload", command=image_prediction)
upload_button.place(relx=0.5, rely=0.5, anchor='center')

result_label = tkinter.Label(text="")
result_label.place(relx=0.5, rely=0.6, anchor='center')



app.mainloop()
