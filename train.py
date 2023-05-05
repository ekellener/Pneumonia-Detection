

from torch.optim import SGD as SGD
from torch.nn import CrossEntropyLoss

import utils
from prepare_train_test_val import generate_trainloaders
from models import chexnet,resnet
from utils import predict,check_gpu
import argparse
from pathlib import Path






train_on_gpu = check_gpu()

#should only be CPU
if not train_on_gpu:
    print('Cuda is not available for traning.Traning on CPU.......')
else:
    print('Cuda is available for traning. Traning on GPU.......')


model_list ={'chexnet': chexnet(),
             'resnet' : resnet(num_of_classes = 2)}

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--lr',type=float,default=0.1)
    arg('--n_epochs',type=int, default = 5)
    arg('--batch-size',type=int, default= 32)
    arg('--data_dir',type=str,default = 'chest_xray')
    arg('--model',type=str, default ='chexnet',choices = model_list.keys())
    arg('--root',type=str,default ='runs/debug', help = 'checkpoint root')

    args = parser.parse_args()

    train_loader= generate_trainloaders(data_dir= args.data_dir,
                                                batch_size= args.batch_size)

    root = Path(args.root)
    root.mkdir(exist_ok = True, parents = True)

    model = model_list[args.model]
    loss = CrossEntropyLoss()

    utils.fit(
        init_optimizer= lambda lr:SGD(model.parameters(),lr = args.lr),
        args= args,
        model = model,
        train_loader = train_loader,
        criterion= loss,
        n_epochs= args.n_epochs,
        train_on_gpu=check_gpu(),
        dir_save = args.root,
        lr = args.lr,
        base_model=args.model
    )

    #y_pred,y_test = predict(model,test_loader,train_on_gpu=check_gpu())

if __name__ == '__main__':
    main()