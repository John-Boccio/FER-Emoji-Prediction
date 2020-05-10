from PIL import Image, ImageTk

import cv2
from FER import fer
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

class App(object):
    def __init__(self, cnn, img_transform):
        self.cnn = cnn
        self.img_transform = img_transform

        self.width = 1400
        self.height = 800
        self.padx = 10
        self.pady = 10
        self.img_width = int((self.width - 4 * self.padx) / 2)
        self.img_height = int(self.img_width * (3 / 4))

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
        self._update_graph([1/len(self.expressions)] * len(self.expressions))

        self.userExpression = tk.StringVar()
        self.userExpression.set("Your expression is: ")

        self.lbl2 = ttk.Label(self.window,textvariable=self.userExpression,
                              font=('cambria', 20, ' bold '))
        self.lbl2.grid(row=1, columnspan=2)

        self.emoji1 = ttk.Label()
        self.emoji1.grid(row=2, columnspan=2,padx=10, pady=10)

        self.find_emojis_button = ttk.Button(self.window, text="Find Emoji's", command=self._find_emojis_event)
        self.find_emojis_button.grid(row=3, columnspan=2, padx=self.padx, pady=self.pady)

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
        plt.cla()
        plt.title("Probabilities For Each Expression")
        plt.xlabel("Expression")
        plt.xticks(rotation=45)
        plt.ylabel("Probability")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        self.ax.bar(self.expressions, probabilities)
        plt.savefig("images/plot.jpg", dpi=1200, quality=100, bbox_inches='tight')
        image = Image.open("images/plot.jpg")
        image = image.resize((self.img_width, self.img_height))
        image_tk = ImageTk.PhotoImage(image)
        self.graph_label.configure(image=image_tk)
        self.graph_label.image = image_tk

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

        self.emoji1.r

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
        self.userExpression.set("Your expression is: " + expression)
        self._update_emojis(expression)
        print(f"Image Captured -\tExpression: {expression:<10}\tProbabilities: {probabilities}")

