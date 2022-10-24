import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os
from torch.nn import Softmax 
import cv2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   Path from dir of model and images
img_p = 'images'
save_p = './output'
model_p = 'checkpoint/stomach_classified_1019.pth'

#   Loading model
model = torch.load(model_p,map_location=device)
model = model.eval()

#   Class labels for prediction
class_names = ['angle', 'duodenum', 'esophagus', 'greater_curvature', 'hypopharnyx', 'junction', 'pylorus', 'reverse']

#   Preprocessing transformations
preprocess=transforms.Compose([
        
        transforms.Resize(size=512),
        transforms.CenterCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

#   Loading image
#   Inference
imgs = os.listdir(img_p)
for i in imgs:
    if i == '.DS_Store': continue
    i_path = os.path.join(img_p, i)
    img = Image.open(i_path).convert('RGB')
    inputs = preprocess(img).unsqueeze(0).to(device)
    outputs = model(inputs)
    soft = Softmax(dim=1)
    out = soft(outputs)
    out = out.cpu().detach().numpy()
    out = out[0]>0.95
    pred = np.where(out==True)[0]
    if len(pred) != 0:
        label = class_names[pred[0]]
    else:
        label = 'None'
    
    img = np.array(img)
    cv2.putText(img, label, (100,100), cv2.FONT_ITALIC, 1, (127, 255, 0), 2, cv2.LINE_AA)
    imgOutputPath = os.path.join(save_p, i)
    cv2.imwrite(imgOutputPath, img)
    


    