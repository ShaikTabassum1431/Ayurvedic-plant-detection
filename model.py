import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import asyncio
import edge_tts
from playsound import playsound
import os 
data_train_path="C:\\Users\\mansu\\AppData\\Local\\Programs\\Python\\Python39\\p2\\archive (6)\\imagedataset\\train"
data_test_path="C:\\Users\\mansu\\AppData\\Local\\Programs\\Python\\Python39\\p2\\archive (6)\\imagedataset\\test"
data_val_path="C:\\Users\\mansu\\AppData\\Local\\Programs\\Python\\Python39\\p2\\archive (6)\\imagedataset\\train"
img_width=180
img_height=180
data_train=tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False)
data_val=tf.keras.utils.image_dataset_from_directory(data_val_path,image_size=(img_width,img_height),
                                               batch_size=
                                                     32,
                                               shuffle=False,

                                               validation_split=False)
data_test=tf.keras.utils.image_dataset_from_directory(data_test_path,image_size=(img_width,img_height),
                                               shuffle=False,
                                                      batch_size=32,

                                               validation_split=False)
data_cat=data_train.class_names
                                                
model=Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(len(data_cat))])
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
epochs_size=25
history=model.fit(data_train,validation_data=data_val,epochs=epochs_size)
epochs_range=range(epochs_size)
#predicting image name
image="C:\\Users\\mansu\\OneDrive\\Desktop\\WhatsApp Image 2024-09-14 at 10.44.39_2aaa27a8.jpg"
image=tf.keras.utils.load_img(image,target_size=(img_height,img_width))
img_arr=tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)
predict=model.predict(img_bat)
score=tf.nn.softmax(predict)
s='plant in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], np.max(score) * 100)
print(s)
"""
#convertion text to speech
async def text_to_speech(text):
    voice = "en-US-AriaNeural"
    tts = edge_tts.Communicate(text, voice)
    await tts.save("output.mp3")
# Run the text-to-speech conversion
asyncio.run(text_to_speech(s))
playsound("output.mp3")
"""
