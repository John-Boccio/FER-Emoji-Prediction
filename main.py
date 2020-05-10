import app
from FER import VggFaceFer

import torch
import torchvision.transforms as transforms


def mult_255(x):
    return x * 255


fer_model = VggFaceFer()
fer_model.eval()
if torch.cuda.is_available():
    fer_model = torch.nn.DataParallel(fer_model.cuda())

vgg_transform = transforms.Compose([transforms.Resize((fer_model.meta["imageSize"][0], fer_model.meta["imageSize"][1])),
                                    transforms.ToTensor(),
                                    transforms.Lambda(mult_255),
                                    transforms.Normalize(mean=fer_model.meta["mean"], std=fer_model.meta["std"])])

app = app.App(fer_model, vgg_transform)
app.run()
app.end()
