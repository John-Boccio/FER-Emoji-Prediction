from PIL import Image, ImageTk

import cv2
from FER import fer
import tkinter as tk
from tkinter import ttk


class App(object):
    def __init__(self, cnn, img_transform):
        self.cnn = cnn
        self.img_transform = img_transform

        self.video_stream = cv2.VideoCapture(0)

        self._init_gui()

    def _init_gui(self):
        self.window = tk.Tk()
        self.window.title("FER Emoji Prediction")
        self.window.geometry('720x480')
        self.face_label = ttk.Label()
        self.face_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.capture_image = ttk.Button(
            self.window,
            text="Capture Image",
            command=self.capture_image_clicked
        )
        self.capture_image.pack(side=tk.BOTTOM, padx=10, pady= 10)

    def run(self):
        self.window.mainloop()

    def end(self):
        self.video_stream.release()

    def capture_image_clicked(self):
        _, image = self.video_stream.read()
        face = fer.find_face(image)
        if face is None:
            tk_img = ImageTk.PhotoImage(Image.fromarray(image))
            self.face_label.configure(image=tk_img)
            print("No face found")
            return

        top, bottom, left, right = face['top'], face['bottom'], face['left'], face['right']
        face_img = image[top:bottom, left:right]
        expression, probabilities = fer.get_expression(self.cnn, self.img_transform(Image.fromarray(face_img)))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, expression, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        tk_img = ImageTk.PhotoImage(Image.fromarray(image))
        self.face_label.configure(image=tk_img)

        print(f"Image Captured -\tExpression: {expression:<10}\tProbabilities: {probabilities}")
