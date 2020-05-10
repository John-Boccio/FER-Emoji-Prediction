from PIL import Image, ImageTk

import cv2
from FER import fer
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


class App(object):
    def __init__(self, cnn, img_transform):
        self.cnn = cnn
        self.img_transform = img_transform

        self.width = 1080
        self.height = 720
        self.padx = 10
        self.pady = 10
        self.img_width = int((self.width - 4*self.padx) / 2)
        img_width_ratio = self.img_width / self.width
        self.img_height = int(self.height * img_width_ratio)

        self.video_stream = cv2.VideoCapture(0)

        self.expressions = [expr.name for expr in fer.FerExpression]

        self._init_gui()

    def _init_gui(self):
        self.window = tk.Tk()
        self.window.title("FER Emoji Prediction")
        self.window.geometry(f'{self.width}x{self.height}')
        # Don't allow any resizing
        self.window.resizable(0, 0)

        self.face_label = ttk.Label()
        self.face_label.grid(row=0, column=0, padx=self.padx, pady=self.pady)
        self._update_face(cv2.imread("images/blank-profile-picture.jpg"))

        self.graph_label = ttk.Label()
        self.graph_label.grid(row=0, column=1, padx=10, pady=10)
        self.fig, self.ax = plt.subplots()
        plt.title("Probabilities For Each Expression")
        plt.xlabel("Expression")
        plt.xticks(rotation=45)
        plt.ylabel("Probability")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        self._update_graph([1/len(self.expressions)] * len(self.expressions))

        self.find_emojis_button = ttk.Button(self.window, text="Find Emoji's", command=self._find_emojis_event)
        self.find_emojis_button.grid(row=1, columnspan=2, padx=self.padx, pady=self.pady)

    def run(self):
        self.window.mainloop()

    def end(self):
        self.video_stream.release()

    def _update_face(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((self.img_width, self.img_height))
        image_tk = ImageTk.PhotoImage(image)
        self.face_label.configure(image=image_tk)
        self.face_label.image = image_tk

    def _update_graph(self, probabilities):
        plt.bar(self.expressions, probabilities)
        plt.savefig("images/plot.jpg", dpi=1200, quality=100)
        image = Image.open("images/plot.jpg")
        image = image.resize((self.img_width, self.img_height))
        image_tk = ImageTk.PhotoImage(image)
        self.graph_label.configure(image=image_tk)
        self.graph_label.image = image_tk

    def _find_emojis_event(self):
        _, image = self.video_stream.read()
        face = fer.find_face(image)
        if face is None:
            self._update_face(image)
            self._update_graph([0.0] * len(self.expressions))
            print("No face found")
            return

        top, bottom, left, right = face['top'], face['bottom'], face['left'], face['right']
        face_img = image[top:bottom, left:right]
        expression, probabilities = fer.get_expression(self.cnn, self.img_transform(Image.fromarray(face_img)))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, expression, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self._update_face(image)
        self._update_graph(probabilities)

        print(f"Image Captured -\tExpression: {expression:<10}\tProbabilities: {probabilities}")
