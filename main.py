from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.uix.image import Image
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array  # Ajout de cette ligne
from keras.applications.mobilenet_v2 import preprocess_input

import os


# Obtenez le chemin absolu du répertoire de données de votre application
data_dir = os.path.abspath("face_detector")

# Utilisez des chemins relatifs depuis le répertoire de données
faceNet = cv2.dnn.readNet(os.path.join(data_dir, "deploy.prototxt"), os.path.join(data_dir, "res10_300x300_ssd_iter_140000.caffemodel"))
maskNet = load_model("mask_detector.model")

class MaskDetectionApp(App):
    def build(self):
        self.camera = Camera(play=True)
        self.camera.resolution = (640, 480)
        self.layout = BoxLayout(orientation="vertical")
        self.layout.add_widget(self.camera)
        self.result_label = Label(text="Mask: Unknown", font_size=24)
        self.layout.add_widget(self.result_label)
        self.capture = None
        Clock.schedule_interval(self.update, 1.0 / 30)
        return self.layout

    def update(self, dt):
        frame = self.camera.texture
        if frame:
            data = frame.pixels
            data = np.frombuffer(data, dtype=np.uint8)
            frame = data.reshape(frame.height, frame.width, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Perform face mask detection here (use your detect_and_predict_mask function)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            label = "Masque:détecté"
            if len(locs) > 0:
                (mask, withoutMask) = preds[0]
                label = "Masque detecté" if mask > withoutMask else "masque non detecté"
                label = f"Masque: {label} ({max(mask, withoutMask) * 100:.2f}%)"
            self.result_label.text = label

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a prediction if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on all
        # faces at the same time rather than one-by-one predictions
        # in the above for loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

if __name__ == '__main__':
    MaskDetectionApp().run()
