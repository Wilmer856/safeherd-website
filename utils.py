################################################################################
#
# Utility functions used by a Django server to process and give results for the Machine
# learning model that predicts Domestic VS Predator animals.
#
################################################################################


import numpy as np
import tensorflow as tf


# Function preprocess_unseen_data(): Preprocesses the image
def preprocess_unseen_data(image_bytes):

    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    return img

# Function post_process_prediction(): Returns the real label by reversing the encoder provided as an argument.


def post_process_prediction(predictions, encoder):
    predicted_labels = (np.max(predictions, axis=1) >= .50).astype(int)
    predicted_labels = encoder.inverse_transform(predicted_labels)
    return predicted_labels.tolist()

# Function get_likelihood(): Uses a threshold to evaluate whether the model predicition is confident or not according to the probability argument.


def get_likelihood(probability, threshold=0.7):
    if probability >= threshold:
        return "Likely"
    else:
        return "Unlikely"

# Function predict_image(): Returns a dictionary containing the prediction results.


def predict_image(image_bytes, interpreter, encoder):
    processed_img = preprocess_unseen_data(image_bytes)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    unseen_data_tensor = np.expand_dims(processed_img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], unseen_data_tensor)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = post_process_prediction(prediction, encoder)
    predicted_prob = np.max(prediction, axis=1).tolist()

    if predicted_prob[0] < 0.50:
        predicted_prob[0] = 1 - predicted_prob[0]

    likelihood = get_likelihood(predicted_prob[0])

    result_dict = {"prediction": predicted_label,
                   "probability": round(predicted_prob[0], 4),
                   "likelihood": likelihood}
    return result_dict
