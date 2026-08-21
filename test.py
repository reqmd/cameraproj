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

import yaml

with open ('models.yaml', 'r') as file:
    data = yaml.safe_load(file)
    RKNN_PATH = data['rknn']
    
from rknnlite.api import RKNNLite
import numpy as np, time

rknn = RKNNLite()
rknn.load_rknn(RKNN_PATH)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

dummy = np.zeros((1, 164, 16, 3), dtype=np.uint8)

# прогрев
for _ in range(5):
    rknn.inference(inputs=[dummy])

# замер 100 вызовов
t0 = time.perf_counter()
for _ in range(100):
    rknn.inference(inputs=[dummy])
print("среднее на вызов:", (time.perf_counter()-t0)/100*1000, "мс")