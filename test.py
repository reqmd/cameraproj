import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
    
def eval(model, image, device = 'cpu'):
    y_preds = []
    model.eval()
    X = X.to(device)
    y = y.to(device)
    y_pred = model(image)
    y_pred = torch.argmax(y_pred, dim=1)
    y_preds.extend(y_pred.cpu().numpy())
    return y_preds