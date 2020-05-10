from enum import Enum
import torch
import cv2
import face_recognition


class FerExpression(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SURPRISE = 5
    NEUTRAL = 6


def get_expression(model, img):
    if torch.cuda.is_available():
        img = img.cuda(non_blocking=True)
    with torch.no_grad():
        model_prediction = model.forward(img.unsqueeze(0))
    _, predicted = torch.max(model_prediction.data, 1)
    expression = FerExpression(predicted.item())
    prob_dist = torch.nn.functional.softmax(model_prediction, dim=1).tolist()[0]
    # Return the expression with the greatest probability and the probability distribution
    return expression.name, prob_dist


def find_face(image):
    downscaled_img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    downscaled_img_rgb = cv2.cvtColor(downscaled_img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(downscaled_img_rgb, model="cnn")
    if len(faces) == 0:
        return None

    top, right, bottom, left = faces[0]
    top *= int(1/.25)
    right *= int(1/.25)
    bottom *= int(1/.25)
    left *= int(1/.25)
    return {
        'top': top,
        'right': right,
        'bottom': bottom,
        'left': left
    }


