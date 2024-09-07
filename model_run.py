#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set the path to your dataset
train_data_dir = './Pistachio_Image_Dataset/'
input_shape = (224, 224)
batch_size = 32
num_classes = 2

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

test_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data



## 3. ResNet50

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Load the pre-trained ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in resnet_model.layers:
    layer.trainable = False

# Add a custom classification head on top of the pre-trained model
flatten = Flatten()(resnet_model.output)
dense1 = Dense(256, activation='relu')(flatten)
output = Dense(1, activation='sigmoid')(dense1)

# Create the final model
model = Model(inputs=resnet_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Train the model
epochs = 20  # Adjust the number of epochs as needed
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs, validation_data=test_generator
)

# Generate predictions for the train data
train_pred = model.predict(train_generator)
train_pred_labels = np.argmax(train_pred, axis=1)  # Assuming the predicted probabilities are in one-hot encoded format

# Collect the true class labels for the train data
train_true_labels = train_generator.classes

# Compute the confusion matrix for the train data
train_cm = classification_report(train_true_labels, train_pred_labels)

# Generate predictions for the test data
test_pred = model.predict(test_generator)
test_pred_labels = np.argmax(test_pred, axis=1)  # Assuming the predicted probabilities are in one-hot encoded format

# Collect the true class labels for the test data
test_true_labels = test_generator.classes

# Compute the confusion matrix for the test data
test_cm = classification_report(test_true_labels, test_pred_labels)

print("Confusion Matrix (Train Data):")
print(train_cm)

print("Confusion Matrix (Test Data):")
print(test_cm)

# Get the training loss and accuracy values from the history object
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']

# Get the validation loss and accuracy values from the history object
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Plot the training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig()

# ## 4. MobileNet

# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model

# # Load the pre-trained MobileNetV2 model
# mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Freeze the layers of the pre-trained model
# for layer in mobilenet_model.layers:
#     layer.trainable = False

# # Add a custom classification head on top of the pre-trained model
# global_avg_pooling = GlobalAveragePooling2D()(mobilenet_model.output)
# dense1 = Dense(256, activation='relu')(global_avg_pooling)
# output = Dense(1, activation='sigmoid')(dense1)

# # Create the final model
# model = Model(inputs=mobilenet_model.input, outputs=output)

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Print the model summary
# print(model.summary())

# # Train the model
# epochs = 20  # Adjust the number of epochs as needed
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=epochs, validation_data=test_generator
# )

# # Generate predictions for the train data
# train_pred = model.predict(train_generator)
# train_pred_labels = np.argmax(train_pred, axis=1)  # Assuming the predicted probabilities are in one-hot encoded format

# # Collect the true class labels for the train data
# train_true_labels = train_generator.classes

# # Compute the confusion matrix for the train data
# train_cm = classification_report(train_true_labels, train_pred_labels)

# # Generate predictions for the test data
# test_pred = model.predict(test_generator)
# test_pred_labels = np.argmax(test_pred, axis=1)  # Assuming the predicted probabilities are in one-hot encoded format

# # Collect the true class labels for the test data
# test_true_labels = test_generator.classes

# # Compute the confusion matrix for the test data
# test_cm = classification_report(test_true_labels, test_pred_labels)

# print("Confusion Matrix (Train Data):")
# print(train_cm)

# print("Confusion Matrix (Test Data):")
# print(test_cm)

# # Get the training loss and accuracy values from the history object
# train_loss = history.history['loss']
# train_accuracy = history.history['accuracy']

# # Get the validation loss and accuracy values from the history object
# val_loss = history.history['val_loss']
# val_accuracy = history.history['val_accuracy']

# # Plot the training and validation loss
# plt.figure(figsize=(8, 6))
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Plot the training and validation accuracy
# plt.figure(figsize=(8, 6))
# plt.plot(train_accuracy, label='Training Accuracy')
# plt.plot(val_accuracy, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# plt.savefig()

# ## 5. Xception

# from tensorflow.keras.applications import Xception
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model

# # Load the pre-trained Xception model
# xception_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Freeze the layers of the pre-trained model
# for layer in xception_model.layers:
#     layer.trainable = False

# # Add a custom classification head on top of the pre-trained model
# global_avg_pooling = GlobalAveragePooling2D()(xception_model.output)
# dense1 = Dense(256, activation='relu')(global_avg_pooling)
# output = Dense(1, activation='sigmoid')(dense1)

# # Create the final model
# model = Model(inputs=xception_model.input, outputs=output)

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Print the model summary
# print(model.summary())

# # Train the model
# epochs = 20  # Adjust the number of epochs as needed
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=epochs, validation_data=test_generator
# )

# # Generate predictions for the train data
# train_pred = model.predict(train_generator)
# train_pred_labels = np.argmax(train_pred, axis=1)  # Assuming the predicted probabilities are in one-hot encoded format

# # Collect the true class labels for the train data
# train_true_labels = train_generator.classes

# # Compute the confusion matrix for the train data
# train_cm = classification_report(train_true_labels, train_pred_labels)

# # Generate predictions for the test data
# test_pred = model.predict(test_generator)
# test_pred_labels = np.argmax(test_pred, axis=1)  # Assuming the predicted probabilities are in one-hot encoded format

# # Collect the true class labels for the test data
# test_true_labels = test_generator.classes

# # Compute the confusion matrix for the test data
# test_cm = classification_report(test_true_labels, test_pred_labels)

# print("Confusion Matrix (Train Data):")
# print(train_cm)

# print("Confusion Matrix (Test Data):")
# print(test_cm)

# # Get the training loss and accuracy values from the history object
# train_loss = history.history['loss']
# train_accuracy = history.history['accuracy']

# # Get the validation loss and accuracy values from the history object
# val_loss = history.history['val_loss']
# val_accuracy = history.history['val_accuracy']

# # Plot the training and validation loss
# plt.figure(figsize=(8, 6))
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Plot the training and validation accuracy
# plt.figure(figsize=(8, 6))
# plt.plot(train_accuracy, label='Training Accuracy')
# plt.plot(val_accuracy, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# plt.savefig()

