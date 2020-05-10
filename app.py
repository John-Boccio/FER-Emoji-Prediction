from PIL import Image, ImageTk

import cv2
from FER import fer
import tkinter as tk
from tkinter import ttk
import numpy as np

from PIL import Image as PImage

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

        blank_profile_picture = ImageTk.PhotoImage(Image.open('images/blank-profile-picture.jpg'))
        self.face_label = ttk.Label(image=blank_profile_picture)
        self.face_label.image = blank_profile_picture
        self.face_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.userExpression = tk.StringVar()
        self.userExpression.set("Your expression is: ")

        self.lbl2 = ttk.Label(self.window,textvariable=self.userExpression,
                        width=200, font=('cambria', 20, ' bold '))
        self.lbl2.pack(side=tk.BOTTOM)

        self.emoji1 = ttk.Label()
        self.emoji1.pack(side=tk.BOTTOM)

        self.capture_image = ttk.Button(
            self.window,
            text="Capture Image",
            command=self._capture_image_clicked
        )
        self.capture_image.pack(side=tk.BOTTOM, padx=10, pady= 10)



    def run(self):
        self.window.mainloop()

    def end(self):
        self.video_stream.release()

    def _update_face(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tk = ImageTk.PhotoImage(Image.fromarray(image_rgb))
        self.face_label.configure(image=image_tk)
        self.face_label.image = image_tk

    def _update_emojis(self, expression):
        if expression == "HAPPY":
            happy = ImageTk.PhotoImage(Image.open("images/emojis/happy/happy.png"))
            self.emoji1.configure(image=happy)
            self.emoji1.image = happy

        elif expression == "ANGRY":
            angry = ImageTk.PhotoImage(Image.open("images/emojis/angry/angry.png"))
            self.emoji1.configure(image=angry)
            self.emoji1.image = angry

        elif expression == "DISGUST":
            disgust = ImageTk.PhotoImage(Image.open("images/emojis/disgust/disgust.png"))
            self.emoji1.configure(image=disgust)
            self.emoji1.image = disgust

        elif expression == "FEAR":
            fear = ImageTk.PhotoImage(Image.open("images/emojis/fear/fear.png"))
            self.emoji1.configure(image=fear)
            self.emoji1.image = fear

        elif expression == "NEUTRAL":
            natural = ImageTk.PhotoImage(Image.open("images/emojis/natural/natural.png"))
            self.emoji1.configure(image=natural)
            self.emoji1.image = natural

        elif expression == "SAD":
            sad = ImageTk.PhotoImage(Image.open("images/emojis/sad/sad.png"))
            self.emoji1.configure(image=sad)
            self.emoji1.image = sad

        elif expression == "SURPRISE":
            surprise = ImageTk.PhotoImage(Image.open("images/emojis/surprise/surprise.png"))
            self.emoji1.configure(image=surprise)
            self.emoji1.image = surprise

    def _capture_image_clicked(self):
        _, image = self.video_stream.read()
        face = fer.find_face(image)
        if face is None:
            self._update_face(image)
            print("No face found")
            return

        top, bottom, left, right = face['top'], face['bottom'], face['left'], face['right']
        face_img = image[top:bottom, left:right]
        expression, probabilities = fer.get_expression(self.cnn, self.img_transform(Image.fromarray(face_img)))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, expression, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self._update_face(image)
        self.userExpression.set("Your expression is: " + expression)
        self._update_emojis(expression)
        print(f"Image Captured -\tExpression: {expression:<10}\tProbabilities: {probabilities}")



