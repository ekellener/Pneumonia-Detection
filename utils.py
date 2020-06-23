
import torch
import os
import json 
from pathlib import Path
from datetime import datetime
from torch.autograd import Variable
import random
import tqdm
import numpy as np 


from models import chexnet,resnet


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x

def check_gpu():
    train_on_gpu = torch.cuda.is_available()

    return train_on_gpu


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()

def get_model(model_path):
    model = chexnet()
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    return model

def fit(args,model=None,criterion = None, init_optimizer= None,train_loader = None, n_epochs = 5,train_on_gpu= False,dir_save='/',lr = 0.1,
        base_model = 'chexnet'):
  
    # after completing your model, if GPU is available, move the model to GPU

    lr = args.lr

    criterion = criterion
    optimizer = init_optimizer(lr)

    #optimizer = optim.SGD(model.parameters(),lr = lr)

    root = Path(args.root)
    model_path = root /'{model}_{lr}_{epoch}'.format(model = base_model,lr = lr, epoch = n_epochs) 

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step  = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch,step))
    else:
        epoch = 1
        step =0

    
    save = lambda ep:torch.save({
        'model' : model.state_dict(),
        'epoch' : ep,
        'step'  : step,
    }, str(model_path))

    
    report_each = 2
    log = root.joinpath('train_{model}.log'.format(model = base_model)).open('at', encoding='utf8')

    if train_on_gpu:
        model.cuda()

    for epoch in range(1,n_epochs+1):

        train_loss = 0.0

        random.seed()
        tq = tqdm.tqdm(total=len(train_loader) * args.batch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch,lr))
        losses = []
        tl = train_loader

        try:
            mean_loss = 0
            for batch_idx, (inputs,targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)
                
                # forward pass : computer predicted outputs by passing inputs to the model
                outputs = model(inputs)

                #calculate the batch loss
                loss = criterion(outputs,targets)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                batch_size = inputs.size(0)

                #backward pass: computer gradient of the loss with respect to the model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()

                step += 1

                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss ='{:.5f}'.format(mean_loss))
                if(batch_idx and batch_idx%report_each ==0):
                    write_event(log,step,loss = mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

def predict(model=None, testloader=None,train_on_gpu= False):

    model.eval()

    correct = 0
    total = 0
    all_predicted = []
    y_test =  []
    with torch.no_grad():
        for images, labels in testloader:

            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model.forward(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if train_on_gpu:
                all_predicted += predicted.cpu().numpy().tolist()
                y_test += labels.cpu().numpy().tolist()
            else:
                all_predicted += predicted.numpy().tolist()
                y_test += labels.numpy().tolist()

    print('Accuracy on test images: %d %%' % (100 * correct / total))

    return all_predicted,y_test