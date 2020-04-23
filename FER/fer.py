from enum import Enum
import torch


class FerExpression(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SURPRISE = 5
    NEUTRAL = 6


def get_expression(model, img, need_softmax=False, exp_class=FerExpression):
    if torch.cuda.is_available():
        img = img.cuda(non_blocking=True)
    with torch.no_grad():
        model_prediction = model.forward(img.unsqueeze(0))
    _, predicted = torch.max(model_prediction.data, 1)
    expression = exp_class(predicted.item())

    if need_softmax:
        prob_dist = torch.nn.functional.softmax(model_prediction, dim=1).tolist()[0]
    else:
        prob_dist = model_prediction.tolist()[0]

    # Return the expression with the greatest probability and the probability distribution
    return expression.name, prob_dist
