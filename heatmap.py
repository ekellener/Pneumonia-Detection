import os
import numpy as np
import time
import sys
from PIL import Image
import utils
import cv2
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms


#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel,transCrop):
       
        #model = torch.nn.DataParallel(model).cuda()

        #modelCheckpoint = torch.load(pathModel)
        #model.load_state_dict(modelCheckpoint['state_dict'])
        model = utils.get_model(pathModel)
        self.model = model.densenet121.features
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        
        self.model.cuda()
        output = self.model(input.cuda())
        
        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map
        
        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.5 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)
        
#-------------------------------------------------------------------------------- 
def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path',type = str, default='runs/debug/chexnet_0.1_5')
    arg('--inp_img_pth',type = str, default='chest_xray/test/PNEUMONIA/person1_virus_6.jpeg')
    arg('--out_img_pth',type = str, default='heatmap.png')

    args = parser.parse_args()
    
    transCrop = 224

    h = HeatmapGenerator(args.model_path, transCrop)
    h.generate(args.inp_img_pth, args.out_img_pth, transCrop)



if __name__ == '__main__':
    main()
