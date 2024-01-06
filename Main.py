import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.applications import EfficientNetB0
import tensorflow as tf

folder_names = ['11', '12', '13', '14', '15', '16']
base_dir = "C:/Users/jeeveth/Downloads/dataset/dataset"

# Load images from all folders
images = []
labels = []
for label, folder_name in enumerate(folder_names):
    folder_path = os.path.join(base_dir, folder_name)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # EfficientNetB0 input size is 224x224
        images.append(img)
        labels.append(label)

# Convert lists to arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Convert labels to one-hot encoding
y_train_cnn = to_categorical(y_train)
y_val_cnn = to_categorical(y_val)
y_test_cnn = to_categorical(y_test)

# Load the pre-trained EfficientNetB0 model
efficientnet_base = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Create a new model by adding custom layers on top of the EfficientNetB0 base
model = Sequential()
model.add(efficientnet_base)
model.add(Flatten())  # Custom flattening layer
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())  # Add BatchNormalization
model.add(Dropout(0.5))  # Add dropout
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())  # Add BatchNormalization
model.add(Dropout(0.5))  # Add dropout
model.add(Dense(len(folder_names), activation='softmax'))  # Output layer with the number of classes

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define a callback to save the best model during training
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)


# Train the model with validation data
history = model.fit(
    x_train, y_train_cnn, epochs=13, batch_size=64,
    validation_data=(x_val, y_val_cnn), callbacks=[checkpoint]
)

test_loss, test_acc = model.evaluate(x_test, y_test_cnn)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Print training and validation loss and accuracy
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

print("Final Training Loss:", train_loss)
print("Final Validation Loss:", val_loss)
print("Final Training Accuracy:", train_accuracy)
print("Final Validation Accuracy:", val_accuracy)

# Plot ROC and AUC curves
y_pred = model.predict(x_test)
n_classes = len(folder_names)

# Plot training accuracy and validation accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("training_validation_accuracy.png")

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Ensure proper spacing between subplots
plt.savefig("training_validation_accuracy_loss.png")


fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_cnn[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc="lower right")
plt.show()

# Save the model as a TFLite file
tflite_model_file = "C:/Users/srava/Downloads/dataset/tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(model)
tflite_model = converter.convert()
with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)
