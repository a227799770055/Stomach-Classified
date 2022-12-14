from genericpath import isdir
import cv2
import time
from cv2 import threshold
import numpy as np
import torch
import torch.nn
from torchvision import models, transforms
import os, sys
import traceback
from PIL import Image
from utils.opt import video_parse_opt

def detector(inputs, model):
    #   Class labels for prediction
    class_names = ['Lesser_curvature', 'Duodenum', 'Esophagus', 'Fundus','Greater_curvature', 
                   'Hypopharnyx', 'EG Junction', 'Pylorus' ] #after 1025 weights
    class_names = ['Lesser_curvature', 'Duodenum', 'Esophagus', 'Greater_curvature', 
                   'Hypopharnyx', 'EG Junction', 'Pylorus','Fundus' ]
    soft = torch.nn.Softmax(dim=1)
    outputs = model(inputs)
    out = soft(outputs)
    out = out.cpu().detach().numpy()
    out = out[0]>0.99
    pred = np.where(out==True)[0]
    
    if len(pred) != 0:
        label = class_names[pred[0]]
    else:
        label = 'blank'
    print(label)
    return label 

def time_counter(start, minutes, seconds):
    seconds = int(time.time() - start) - minutes * 60
    if seconds >= 60:
        minutes += 1
        seconds = 0
    mins = "%02d" %minutes
    secs = "%02d" %seconds
    return mins, secs


if __name__ == '__main__':

    opt = video_parse_opt()

    video_path = opt.video_path
    model_path = opt.model_path
    save_path = opt.save_path
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    stomachIcon = cv2.imread('graystomach.png')
    stomachIcon = cv2.resize(stomachIcon, (300, 300), interpolation=cv2.INTER_CUBIC)
    precentLocation = 'blank'
    
    partitionImgURL = {'Lesser_curvature':'partition/lesser curvature.png', 
                'Duodenum':'partition/Duodenum.png', 
                'Esophagus':'partition/Esophagus.png', 
                'Greater_curvature':'partition/Greater curvature.png',
                'Hypopharnyx':'partition/Hypopharnyx.png', 
                'EG Junction':'partition/EG junction.png', 
                'Pylorus':'partition/Pylorus.png', 
                'Fundus':'partition/Fundus.png',
                'blank':'partition/Blank.png'
                }
    partitionImg = {}
    for key in partitionImgURL.keys():
        key_img = cv2.imread(partitionImgURL[key])
        key_img = cv2.resize(key_img, (300, 300), interpolation=cv2.INTER_CUBIC)
        partitionImg[key] = key_img

    #   load model 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device)
    model = model.eval()


    #   frame process
    preprocess=transforms.Compose([
        transforms.Resize(size=512),
        transforms.CenterCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    #   loading video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_name = (video_path.split('/')[-1]).split('.')[0]
    out = cv2.VideoWriter(os.path.join(save_path, '{}.mp4'.format(video_name)), fourcc, fps, (width+300, height))

    frameID = 0
    time_start = time.time()
    seconds, minutes = 0, 0
    
    frame_label = {'Lesser_curvature':0, 'Duodenum':0, 'Esophagus':0, 'Greater_curvature':0, \
                    'Hypopharnyx':0, 'EG Junction':0, 'Pylorus':0, 'Fundus':0, 'blank':0}
    frame_count = {'Lesser_curvature':0, 'Duodenum':0, 'Esophagus':0, 'Greater_curvature':0, \
                    'Hypopharnyx':0, 'EG Junction':0, 'Pylorus':0, 'Fundus':0, 'blank':0}
    timer_label = {'Lesser_curvature':0, 'Duodenum':0, 'Esophagus':0, 'Greater_curvature':0, \
                    'Hypopharnyx':0, 'EG Junction':0, 'Pylorus':0, 'Fundus':0, 'blank':0}
    threshold =  {'Lesser_curvature':False, 'Duodenum':False, 'Esophagus':False, 'Greater_curvature':False, \
                    'Hypopharnyx':False, 'EG Junction':False, 'Pylorus':False, 'Fundus':False, 'blank':False}
    
    label_position = {'Lesser_curvature':[50,350], 'Duodenum':[50,300], 'Esophagus':[50,100], 'Greater_curvature':[50,200], \
                    'Hypopharnyx':[50,50], 'EG Junction':[50,150], 'Pylorus':[50,250], 'Fundus':[50,400]}
    
    image_position = {'Lesser_curvature':[93, 162, 145, 179], 'Duodenum':[45,270, 125, 300], 'Esophagus':[0,23,85,50],
                     'Greater_curvature':[140,275, 300, 300], 'Hypopharnyx':[190,0,295,25], 
                     'EG Junction':[31,98,104,117], 'Pylorus':[12,117,82,201], 'Fundus':[220,35,287,55]}

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            frame_copy = frame.copy()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            frame_copy = Image.fromarray(frame_copy)
            inputs = preprocess(frame_copy).unsqueeze(0).to(device)
            label = detector(inputs, model)
            # frame ????????????
            frame = cv2.copyMakeBorder(frame, 0, 0, 0, 300, cv2.BORDER_CONSTANT,value=(0,0,0))
            blk = np.zeros(frame.shape, np.uint8)  
            # Time counter ??????????????????
            mins, secs = time_counter(time_start, minutes, seconds)
            timestamp = "{}:{}".format(mins, secs)
            cv2.putText(frame, timestamp, (600, 50), cv2.FONT_ITALIC, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # ???????????? label 
            for key in label_position:
                cv2.putText(frame, key, (label_position[key][0], label_position[key][1]), cv2.FONT_ITALIC, 
                            1, (128,128,128), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (width, height), cv2.FONT_ITALIC, 
                            1, (128,128,128), 2, cv2.LINE_AA)
            
            
            # ??????label??????????????????????????????????????????????????????0
            # frame_label ????????????????????????
            # frame_count ????????????????????????
            if frame_label[label]==0:
                frame_label[label]=frameID
            elif frame_label[label] == frameID-1:
                frame_label[label]=frameID
                frame_count[label] = frame_count[label]+1
            elif frame_label[label] != frameID-1:
                frame_count[label] = 0
                frame_label[label] = 0

            # ?????????label?????????????????????????????????????????????label?????????????????????true
            if label != 'blank':
                if threshold[label]==False and frame_count[label]>=20: # threshold setting as 60 frames
                    threshold[label]=True

            # ?????????????????????????????????????????????label
            # ????????????????????????????????????????????????
            labels_key = threshold.keys()
            for key in labels_key:
                if threshold[key] == True:
                    timer_label[key] = timer_label[key] + 1    
                    cv2.putText(frame, key, (label_position[key][0], label_position[key][1]), cv2.FONT_ITALIC, 
                                1, (127, 255, 0), 2, cv2.LINE_AA)  

            # ?????????????????????
            # ??????????????????????????? n frames ?????????????????????
            if label != 'blank' and frame_count[label] >= 20:
                precentLocation = label
            if precentLocation != 'blank':
                frame[250:550, width:width+300] = partitionImg[precentLocation]
            else:
                frame[250:550, width:width+300] = partitionImg['blank']


            frameID += 1
            # ????????????
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            error_class = e.__class__.__name__ #??????????????????
            detail = e.args[0] #??????????????????
            cl, exc, tb = sys.exc_info() #??????Call Stack
            lastCallStack = traceback.extract_tb(tb)[-1] #??????Call Stack?????????????????????
            fileName = lastCallStack[0] #???????????????????????????
            lineNum = lastCallStack[1] #?????????????????????
            funcName = lastCallStack[2] #???????????????????????????
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(errMsg)
            break

    cap.release()
    cv2.destroyAllWindows()
