import cv2
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

camera = cv2.VideoCapture(0)

while True:

    status,frame = camera.read()

    # Modify the input data by:
    if status:
        frame=cv.flip(frame,1)
    # 1. Resizing the image


        resized_frame = cv2.resize(frame,(224,224))
        resized_frame=np.expand_dims(resized_frame,axis=0)

    # 2. Converting the image into Numpy array and increase dimension

   

    # 3. Normalizing the image
        resized_frame = resized_frame/255.0

    # Predict Result
        prediction = model.predict(resized_frame)

        rock=int(prediction[0][0]*100)
        paper=int(prediction[0][1]*100)
        sissor=int(prediction[0][2]*100)
        print(f"rock:{rock}%,paper:{paper}%,sissor:{sissor}%")
        
     
        cv2.imshow("Result",frame)
            
        key = cv2.waitKey(1)

        if key == 32:
        
            break

camera.release()
cv2.destroyAllWindows()