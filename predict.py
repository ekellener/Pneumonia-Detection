import utils
import argparse
from pathlib import Path
from PIL import Image 
import os

import torch
from torchvision import transforms
from torch.autograd import Variable 

def predict_image(model,image_path):
    model.eval()
    train_on_gpu = utils.check_gpu()
    print("Prediction in progress")
    image = Image.open(image_path).convert('RGB')

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)

    

    transformation  = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)


    # Turn the input into a Variable
    input = Variable(image_tensor)
    #should ignore 
    if torch.cuda.is_available():
        input = input.cuda()


    # Predict the class of the image
    output = model.forward(input)
    
    _, predicted = torch.max(output.data, 1)
    if train_on_gpu:
        all_predicted = predicted.cpu().numpy().tolist()
        #y_test += labels.cpu().numpy().tolist()
    else:
        all_predicted = predicted.numpy().tolist()
        #y_test += labels.numpy().tolist()
    return all_predicted[0]
    '''
    #index = output.cpu().data.numpy().argmax()
    print(output.cpu())
    index = output.cpu().data.numpy().argmax()
    return index
    '''

def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path',type = str, default='runs/debug/chexnet_0.1_5')
    #arg('--image_path',type = str, default='chest_xray/test/PNEUMONIA/person1_virus_6.jpeg')
    arg('--image_path',type = str, default='chest_xray/test/NORMAL/NORMAL2-IM-0060-0001.jpeg')

    args = parser.parse_args()

    model = utils.get_model(str(Path(args.model_path)))

    class_label  = predict_image(model,args.image_path)
    
    if(class_label == 1):
        print("The person is diagonised with Pneumonia")
    else:
        print("The person is not affected with Pneumonia")
    '''
    print(os.listdir('chest_xray/test/NORMAL/'))
    

    model = utils.get_model(str(Path('runs/debug/chexnet_0.1_5')))

    files = os.listdir('chest_xray/test/NORMAL/')
    img_dir = 'chest_xray/test/NORMAL/'
    for img_name in files:
        img_path = img_dir + img_name
        print(img_name,predict_image(model,img_path))
        #print('\n')
    '''

if __name__ == '__main__':
    main()