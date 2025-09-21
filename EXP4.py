# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

# Assuming your dataset paths
anshtanwar_jellyfish_types_path = "/kaggle/working/jellyfish_dataset"  # Update this to your downloaded path

# -----------------------------
# Data Preparation
# -----------------------------
def DataCreation(train_dir, test_dir, val_dir):
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=32
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=32
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=32
    )
    return train_generator, test_generator, val_generator

train_generator, test_generator, val_generator = DataCreation(
    os.path.join(anshtanwar_jellyfish_types_path, 'Train_Test_Valid/Train'),
    os.path.join(anshtanwar_jellyfish_types_path, 'Train_Test_Valid/test'),
    os.path.join(anshtanwar_jellyfish_types_path, 'Train_Test_Valid/valid')
)

# -----------------------------
# CNN Model
# -----------------------------
def CreatingCNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = CreatingCNN()
cnn_model.summary()

history = cnn_model.fit(train_generator, epochs=15, validation_data=test_generator)

plt.plot(history.history['accuracy'], label='CNN training accuracy')
plt.plot(history.history['val_accuracy'], label='CNN val accuracy')
plt.legend()
plt.show()

cnn_model.save('/kaggle/working/my_cnn_model.h5')

# -----------------------------
# MLP Model
# -----------------------------
# Flatten images and scale
def LoadImagesForMLP(generator):
    images = []
    labels = []
    for i in range(len(generator)):
        x, y = generator[i]
        images.append(x)
        labels.append(y)
    X = np.vstack(images)
    y = np.vstack(labels)
    return X.reshape(X.shape[0], -1), y

X_train, y_train = LoadImagesForMLP(train_generator)
X_test, y_test = LoadImagesForMLP(test_generator)

def CreatingMLP(input_shape):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

mlp_model = CreatingMLP(X_train.shape[1])
mlp_model.summary()

mlp_history = mlp_model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)

plt.plot(mlp_history.history['accuracy'], label='MLP training accuracy')
plt.plot(mlp_history.history['val_accuracy'], label='MLP val accuracy')
plt.legend()
plt.show()

mlp_model.save('/kaggle/working/my_mlp_model.h5')
