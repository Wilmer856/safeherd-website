import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
import albumentations as Alb
import joblib
from tensorflow import keras
from keras import layers
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
AUTO = tf.data.experimental.AUTOTUNE

labelImageMap = {}
labels = []
images = []

# Path will be the path containing all of the animal images divided between a domestic and predator folder
path = "newAnimals/"
# Append the two labels into the 'labels' list
with open('binaryTest.txt', 'r') as animalFile:
    for animalType in animalFile:
        labels.append(animalType.strip())

# Obtain the animal images and set them as keys in the dictionary with the value being the label for the image
for label in labels:
    for animalFile in os.listdir(os.path.join(path, label)):
        animalFile = os.path.join(path, label, animalFile)
        images.append(animalFile)
        labelImageMap[animalFile] = label

# Turn the dictionary into a Pandas DataFrame
df = pd.DataFrame(index=labelImageMap.keys(), data=labelImageMap.values())
df.rename(columns={0: 'type'}, inplace=True)
df.index.name = 'id'

# Transform labels to numerical values to make the machine learning algirthms work later
encoder = LabelEncoder()
df['type'] = encoder.fit_transform(df['type'])


# Create features and target list and split the data into train/test sets
features = df.index
target = df['type']

X_train, X_value, Y_train, Y_value = train_test_split(
    features, target, test_size=0.15, random_state=10)

# Have various forms of our image in training set like the image being flipped or color changed
transforms_train = Alb.Compose([
    Alb.VerticalFlip(p=0.5),
    Alb.HorizontalFlip(p=0.5),
    Alb.CoarseDropout(p=0.7),
    Alb.RandomGamma(p=0.7),
    Alb.RandomBrightnessContrast(p=0.7)
])

# Function albFunction(): The image will be transformed to various forms using the transforms_train variable


def albFunction(img):
    albData = transforms_train(image=img)
    albImg = albData['image']
    return albImg

# Ensures the image goes through the various forms of the image


@tf.function
def process_data(img, lbl):
    albImg = tf.numpy_function(albFunction, [img], Tout=tf.float32)
    return img, lbl


# Function decode_image(): Reads image and resizes it
# Returns a one-hot vector that represents each categorical value in binary format
def decode_image(image_path, label):
    # Read and decode the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # assuming RGB images

    # Resize the image to [224, 224]
    img = tf.image.resize(img, [224, 224])

    # Normalize pixel values to be between 0 and 1
    img = img / 255.0

    return img, label


# Seperate data into training and validation data to ensure our test/validation data does not get looked at in filtered formats
train_data = (
    tf.data.Dataset
    .from_tensor_slices((X_train, Y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(partial(process_data), num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

val_data = (
    tf.data.Dataset
    .from_tensor_slices((X_value, Y_value))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

# Create a variable containing the pre-trained weight Inception model that is trained with the dataset imagenet.
pre_trained_model = InceptionV3(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False
)

# Set the layers.trainable field to False due to us not needing to retrain the Inception model.
for layer in pre_trained_model.layers:
    layer.trainable = False

# The 'mixed7' layer is a clever/special layer that understand many features of an image
mixedSeven = pre_trained_model.get_layer('mixed7')
mixedSevenOutput = mixedSeven.output


# -------------Implement a model using Keras-------------

# flatten the 'mixed7' output into a one-dimensional array to change the layers from being convolutional to fully connected layers in a neural network
arch = layers.Flatten()(mixedSevenOutput)
# Connect each neuron from one layer to the previous layer and perform a weighted sum of its inputs and perform the activation function 'relu'
arch = layers.Dense(512, activation='relu')(arch)
# Normalize the data to ensure the outputs are not too high or not too low
arch = layers.BatchNormalization()(arch)
arch = layers.Dense(256, activation='relu')(arch)

# Prevent overfitting by dropping 30% of the neurons. This ensures that our model doesn't rely too much on specific features in the training set
arch = layers.Dropout(0.3)(arch)
arch = layers.BatchNormalization()(arch)
# Create the final output of the layers by converting all of the neurons into probabilistic scores
output = layers.Dense(1, activation='sigmoid')(arch)
model = keras.Model(pre_trained_model.input, output)

# Configure the learning process of the model
# 'optimizer' is the algorithm of choice to update model weights when training.
# 'loss' is the objective function that the model is trying to minimize while training. Our choice is typically used for multi-class classification tasks.
# 'metrics' is the metric that will be evaluated by the model during the training and testing process
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# A way to stop the training process due to the validation accuracy reaching a point where the model is working well


class modelCallback(tf.keras.callbacks.Callback):
    def epoch_end(self, epoch, hist={}):
        if hist.get('val_accuracy') > 0.99:
            print(
                '\nValidation accuracy has reached upto 90%\ so, stopping further training.')
            self.model.stop_training = True


# Stops the training accuracy if there's no sign of the validation accuracy improving to stop it from possibly getting slightly worse
earlyStop = EarlyStopping(
    patience=5, monitor='val_accuracy', restore_best_weights=True)

# Reduces the learning rate during training if the validation loss metric is not improving
learingRate = ReduceLROnPlateau(
    monitor='val_loss', patience=2, factor=0.5, verbose=1)

# Trains the model
trained = model.fit(train_data, validation_data=val_data, epochs=10,
                    verbose=1, callbacks=[earlyStop, learingRate, modelCallback()])

# Obtain the history of the training process and turn it into a Pandas DataFrame.
# Uses the history to create two curve plots: loss/val_loss and accuracy/val_accuracy
trainedDf = pd.DataFrame(trained.history)
trainedDf.loc[:, ['loss', 'val_loss']].plot()
trainedDf.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

# ----------------------------- Saving the model as a file -----------------------------

# model.save("animalIDModel.keras")
# joblib.dump(encoder, "animal_encoder_test_1.joblib")

# ----------------------------- Loading the model to save as a tflite file -----------------------------

# model = load_model("animalIDModel_test.keras")
# model.save("saved_model")
# Convert the SavedModel to TensorFlow Lite format
# converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
# tflite_model = converter.convert()

# # Save the TensorFlow Lite model to a file
# with open("animalIDModel_lite.tflite", "wb") as f:
#     f.write(tflite_model)
