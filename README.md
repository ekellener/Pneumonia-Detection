# Pneumonia-Detection
* Analyze chest xrays for diagonising pneumonia and locate the regions with heat map.
* In this work, I used a Transfer Learning Approach to diagnose Pneumonia, by fine-tuning the CheXNet a densenet model developed by [Stanford ML Group](https://stanfordmlgroup.github.io/projects/chexnet/)

# Data 
Download chestXray data from [here](https://data.mendeley.com/datasets/rscbjbr9sj/2) to chest_xray folder

# Instructions to run
User can run each of the tasks as simple python scripts, with different options as command line arguments.Below are the sample command line executions for each tasks

 **Training:** 

     python train.py --batch-size=64 --data_dir='CHEST_XRAYS/' --lr=0.01

**Evaluation:** 

	  python evaluation.py --batch-size=64 --data_dir='chest-xray/' --model_path=runs/debug/chexnet_0.01_5

**Testing on Single Image:** 

    python predict.py --model_path=runs/debug/chexnet_0.01_5 --image_path='chest_xrat/test/NORMAL/NORMAL2-IM-0317-0001.jpeg'
 
 **Generating Heat Map:** 

    python heatmap.py --model_path=runs/debug/chexnet_0.01_5 --inp_img_pth='chest_xray/test/PNEUMONIA/person59_virus_116.jpeg' --out_img_pth='heatmap.png'

# Results and Conclusions:

Densenet, Resnet and CheXnet were experimented for transfer learning, among which CheXnet performed better with 92 % accuracy on test data.

Input Image            |  heatmap          
:-------------------------:|:-------------------------:
![](results/person59_virus_116.jpeg = 250x250)  |  ![](results/heatmap.png = 250x250) 
* This chest xray is of the person with pneumonia
