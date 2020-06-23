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

# Results:

Densenet, Resnet and CheXnet were experimented for transfer learning, among which CheXnet performed better with 92 % accuracy on test data.

#### LBP
Query Image            |  Search Result 1              | Search Result 2
:-------------------------:|:-------------------------:|:-------------------------:
![](queries/Hand_0008111.jpg)  |  ![](results/lbp/Hand_0008110.jpg)  |  ![](results/lbp/Hand_0008130.jpg)
* LBP might have taken color into considerations rather than face of the image

# Conclusions:
* The intricacies of a dealing with vector based representation of multimedia objects was observed in areas spanning that can help retrieve results based on distance measures. 
* Eﬃcient representation of multimedia objects using feature vectors using feature descriptors-LBP and SIFT was implemented, storing the obtained results in a ﬁle system and to see how it helps the ease the process of comparison.
* For now, cosine similarity was used to measure similarity between LBP and FLANN Matcher was used to match the keypoints in SIFT.
* The results can be further improved by experimenting with different similarity metrics like chi square distance.
