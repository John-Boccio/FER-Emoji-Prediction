from FER import VggFaceFer, get_expression
from PIL import Image

import cv2
import face_recognition
import numpy as np
import torch
import torchvision.transforms as transforms


def mult_255(x):
    return x * 255


fer_model = VggFaceFer()
fer_model.eval()
if torch.cuda.is_available():
    fer_model = torch.nn.DataParallel(fer_model.cuda())

vgg_transform = transforms.Compose([transforms.Resize(fer_model.meta["imageSize"][0]),
                                    transforms.ToTensor(),
                                    transforms.Lambda(mult_255),
                                    transforms.Normalize(mean=fer_model.meta["mean"], std=fer_model.meta["std"])])

video_stream = cv2.VideoCapture(0)

while True:
    _, f = video_stream.read()

    # Downscale image so it can run through face_recognition faster
    downscaled_f = cv2.resize(f, (0, 0), fx=0.2, fy=0.2)
    downscaled_f_rgb = cv2.cvtColor(downscaled_f, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(downscaled_f_rgb)
    for top, right, bottom, left in faces:
        # Reverse the downscaling so we can display on full resolution image
        top *= int(1/.2)
        right *= int(1/.2)
        bottom *= int(1/.2)
        left *= int(1/.2)

        face_img = Image.fromarray(f[top:bottom, left:right])
        face_img_transformed = vgg_transform(face_img)
        expression, distribution = get_expression(fer_model,  face_img_transformed, need_softmax=True)

        cv2.rectangle(f, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(f, expression, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"Expression: {expression:15}\tProbability Distribution: {distribution}")

    cv2.imshow('FER', f)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

video_stream.release()
cv2.destroyAllWindows()
