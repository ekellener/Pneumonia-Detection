# Pneumonia-Detection
* Analyze chest xrays for diagonising pneumonia and locate the regions with heat map.
* In this work, I used a Transfer Learning Approach to diagnose Pneumonia, by fine-tuning the CheXNet a densenet model developed by [Stanford ML Group](https://stanfordmlgroup.github.io/projects/chexnet/)



# Files
* feature_descriptors.py - Python file which class implementation of SIFT and LBP
* task1.py-  Python script for Task1
* index.py - Python script for Task2
* searcher.py - Python script with class implementation of similarity and matching
* search.py - Python script for Task3
* small_db - folder with small set of images
* results/lbp - folder where the search results will be stored for lbp descriptor
* results/sift - folder where the search  results will be stored for sift descriptor
* Report.pdf - Report

# Requirments
The system successfully runs on an windows and linux machines with Anaconda3. Programs leverage use of the pandas package for Dataframes, numpy for the arrays, to ensure the direct functional use of feature descriptors i.e,SIFT and LBP we used packages directly available from the opencv-python and scikit-image libraries respectively. Therefore, one needs to install the following packages:


 **Run the following commands:**
 
     conda create -n myenv python=3.6
     conda activate myenv
     conda install -c menpo opencv
     conda install -c anaconda scipy
     conda install -c anaconda numpy

# Instructions to run
User can run each of the tasks as simple python scripts, with different options as command line arguments.Below are the sample command line executions for each tasks

 **Training:** 

     python train.py --batch-size=64 --data_dir='CHEST_XRAYS/' --lr=0.01

**Evaluation:** 

	  python evaluation.py --batch-size=64 --data_dir='CHEST_XRAYS/' --model_path=runs/debug/chexnet_0.01_5

**Testing on Single Image:** 

    python predict.py --model_path=runs/debug/chexnet_0.01_5 --image_path='CHEST_XRAYS/test/NORMAL/NORMAL2-IM-0317-0001.jpeg'
 
 **Generating Heat Map:** 

    python heatmap.py --model_path=runs/debug/chexnet_0.01_5 --inp_img_pth='CHEST_XRAYS/test/PNEUMONIA/person59_virus_116.jpeg' --out_img_pth='heatmap.png'

# Results:

#### LBP
Query Image            |  Search Result 1              | Search Result 2
:-------------------------:|:-------------------------:|:-------------------------:
![](queries/Hand_0008111.jpg)  |  ![](results/lbp/Hand_0008110.jpg)  |  ![](results/lbp/Hand_0008130.jpg)
* LBP might have taken color into considerations rather than face of the image
#### SIFT
Query Image            |  Search Result 1              | Search Result 2
:-------------------------:|:-------------------------:|:-------------------------:
![](queries/Hand_0008111.jpg)  |  ![](results/sift/Hand_0008110.jpg)  |  ![](results/sift/Hand_0011498.jpg)
* SIFT looks for key points and might have taken face of the image into major consideration.

# Conclusions:
* The intricacies of a dealing with vector based representation of multimedia objects was observed in areas spanning that can help retrieve results based on distance measures. 
* Eﬃcient representation of multimedia objects using feature vectors using feature descriptors-LBP and SIFT was implemented, storing the obtained results in a ﬁle system and to see how it helps the ease the process of comparison.
* For now, cosine similarity was used to measure similarity between LBP and FLANN Matcher was used to match the keypoints in SIFT.
* The results can be further improved by experimenting with different similarity metrics like chi square distance.
