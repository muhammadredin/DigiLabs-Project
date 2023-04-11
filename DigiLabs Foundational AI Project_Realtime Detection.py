import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained Keras model
model = load_model('model.h5')

# Define the labels for the two classes
labels = ['unready', 'ready']

# Open the video stream or camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream or camera
    ret, frame = cap.read()

    # Resize the frame to the input size of the model
    input_size = model.input_shape[1:3]
    resized_frame = cv2.resize(frame, input_size)

    # Preprocess the input frame
    preprocessed_frame = np.expand_dims(resized_frame, axis=0) / 255.0

    # Predict the class probabilities for the input frame
    predictions = model.predict(preprocessed_frame)[0]

    # Get the index of the predicted class
    predicted_class = np.argmax(predictions)

    # Get the label of the predicted class
    predicted_label = labels[predicted_class]

    # Add the predicted label to the frame
    cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Wheat Detector', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream or camera and close all windows
cap.release()
cv2.destroyAllWindows()
