from tensorflow import keras
import tensorflow as tf
import cv2 as cv
import numpy as np

model = keras.applications.InceptionV3(weights="imagenet", include_top=True)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Process frame here
    img = tf.image.resize(frame, [299, 299])
    img_tensor = tf.expand_dims(img, 0) # Create a batch
    img_tensor = tf.keras.applications.inception_v3.preprocess_input(img_tensor)
    preds = model(img_tensor)
    preds = tf.make_tensor_proto(preds)
    preds = tf.make_ndarray(preds)
    preds = tf.keras.applications.inception_v3.decode_predictions(preds, top=3)

    Width = frame.shape[1]
    Height = frame.shape[0]
    x = 100
    y = 100
    for pred in preds[0]:
        pred_string = pred[1] + ", " + str(pred[2])
        cv.putText(frame, text=pred_string, org=(y,x), 
            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,0,0))
        x += 50
        y += 50

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
