import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
    
def eval(model, image, device = 'cpu', classes = {0:'Скрепка', 1:'Зерно'}):
    y_preds = []
    model.eval()
    y_pred = model(torch.unsqueeze(image, 0))
    y_pred = torch.argmax(y_pred, dim=1)
    y_preds.extend(y_pred.cpu().numpy())
    return classes[int(y_preds[0])]