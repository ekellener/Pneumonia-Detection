
#from chexnet.DensenetModels import DenseNet121
#import matplotlib.pyplot as plt
#from torch.autograd import Variable
#from PIL import Image
#import torch

from torch.optim import SGD as SGD
from torch.nn import CrossEntropyLoss

import torch
import utils
from prepare_train_test_val import generate_testloader
from models import chexnet,resnet
from utils import predict,check_gpu
import argparse
from pathlib import Path


train_on_gpu = check_gpu()

if not train_on_gpu:
    print('Cuda is not available for testing.Testing on CPU.......')
else:
    print('Cuda is available for testing. Testing on GPU.......')

def get_model(model_path):
    model = chexnet()
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    #arg('--model_path',type = str, default='checkpoints/finetune_chexnet_0.01_3')
    arg('--data_dir',type=str,default = 'chest_xray')
    arg('--batch-size',type = int, default= 32)
    arg('--model_path',type = str, default='runs/debug/chexnet_0.1_5')

    args = parser.parse_args()

    model = utils.get_model(str(Path(args.model_path)))
    test_loader = generate_testloader(data_dir= args.data_dir,
                                                batch_size= args.batch_size)
    y_pred, y_test = utils.predict(
        model = model,
        testloader = test_loader,
        train_on_gpu= check_gpu()
    )
    print(y_pred)
    print(y_test)

    #y_pred,y_test = predict(model,test_loader,train_on_gpu=check_gpu())

if __name__ == '__main__':
    main()