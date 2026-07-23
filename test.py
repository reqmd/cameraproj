import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def eval(model, data, device = 'cpu'):
    y_preds = []
    model.eval()
    for X, _ in data:
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        y_pred = torch.argmax(y_pred, dim=1)
        y_preds.extend(y_pred.cpu().numpy())
    return y_preds