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
    class_names = ['angle', 'duodenum', 'esophagus', 'greater_curvature', 'hypopharnyx', 'junction', 'pylorus', 'fundus']
    soft = torch.nn.Softmax(dim=1)
    outputs = model(inputs)
    out = soft(outputs)
    out = out.cpu().detach().numpy()
    out = out[0]>0.95
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
    
    stomachIcon = cv2.imread('stomachicon.jpg')
    stomachIcon = cv2.resize(stomachIcon, (200, 200), interpolation=cv2.INTER_CUBIC)

    #   load model 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path,map_location=device)
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
    out = cv2.VideoWriter(os.path.join(save_path, '{}.mp4'.format(video_name)), fourcc, fps, (width, height))

    frameID = 0
    time_start = time.time()
    seconds, minutes = 0, 0
    
    frame_label = {'angle':0, 'duodenum':0, 'esophagus':0, 'greater_curvature':0, \
                    'hypopharnyx':0, 'junction':0, 'pylorus':0, 'fundus':0, 'blank':0}
    frame_count = {'angle':0, 'duodenum':0, 'esophagus':0, 'greater_curvature':0, \
                    'hypopharnyx':0, 'junction':0, 'pylorus':0, 'fundus':0, 'blank':0}
    timer_label = {'angle':0, 'duodenum':0, 'esophagus':0, 'greater_curvature':0, \
                    'hypopharnyx':0, 'junction':0, 'pylorus':0, 'fundus':0, 'blank':0}
    threshold =  {'angle':False, 'duodenum':False, 'esophagus':False, 'greater_curvature':False, \
                    'hypopharnyx':False, 'junction':False, 'pylorus':False, 'fundus':False, 'blank':False}
    
    label_position = {'angle':[50,350], 'duodenum':[50,300], 'esophagus':[50,100], 'greater_curvature':[50,200], \
                    'hypopharnyx':[50,50], 'junction':[50,150], 'pylorus':[50,250], 'fundus':[50,400]}
    
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            frame_copy = frame.copy()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            frame_copy = Image.fromarray(frame_copy)
            inputs = preprocess(frame_copy).unsqueeze(0).to(device)
            label = detector(inputs, model)
            
            # Time counter 計算影片時長
            mins, secs = time_counter(time_start, minutes, seconds)
            timestamp = "{}:{}".format(mins, secs)
            cv2.putText(frame, timestamp, (600, 50), cv2.FONT_ITALIC, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 將 stomach icon 與 frame 融合
            frame[height-200:height, width-200:width] = stomachIcon
            
            
            # 加上灰色 label
            for key in label_position:
                cv2.putText(frame, key, (label_position[key][0], label_position[key][1]), cv2.FONT_ITALIC, 
                            1, (128,128,128), 2, cv2.LINE_AA)
            
            # 計算label連續出現的幀數，當沒有連貫時，幀數歸0
            # frame_label 紀錄上一幀的位置
            # frame_count 紀錄連續出現幾幀
            if frame_label[label]==0:
                frame_label[label]=frameID
            elif frame_label[label] == frameID-1:
                frame_label[label]=frameID
                frame_count[label] = frame_count[label]+1
            elif frame_label[label] != frameID-1:
                frame_count[label] = 0
                frame_label[label] = 0

            # 當特定label的幀數連續出現且達到閥值時，該label的閥門就會變成true
            if label != 'blank':
                if threshold[label]==False and frame_count[label]>30: # threshold setting as 60 frames
                    threshold[label]=True

            # 當閥門打開後，就會顯示偵測到該label
            # 當顯示一定的時間後，閥門會在關閉
            labels_key = threshold.keys()
            for key in labels_key:
                if threshold[key] == True:
                    timer_label[key] = timer_label[key] + 1    
                    cv2.putText(frame, key, (label_position[key][0], label_position[key][1]), cv2.FONT_ITALIC, 
                                1, (127, 255, 0), 2, cv2.LINE_AA)  

            frameID += 1
            # 寫入影片
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            error_class = e.__class__.__name__ #取得錯誤類型
            detail = e.args[0] #取得詳細內容
            cl, exc, tb = sys.exc_info() #取得Call Stack
            lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
            fileName = lastCallStack[0] #取得發生的檔案名稱
            lineNum = lastCallStack[1] #取得發生的行號
            funcName = lastCallStack[2] #取得發生的函數名稱
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(errMsg)
            break

    cap.release()
    cv2.destroyAllWindows()
