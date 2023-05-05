from torchvision import datasets, transforms
import torch
import os



def generate_trainloaders(batch_size  = 32, num_workers = 0, data_dir = 'chest_xray'):
        
    train_dir = os.path.join(data_dir,'train/')
    # creating train and test data using Image Folder

    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    train_data = datasets.ImageFolder(train_dir,transform=data_transform)
    print('Number of training images: ', len(train_data))
    train_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,
                                          num_workers = num_workers, shuffle = True)


    return train_loader

def generate_testloader(batch_size  = 32, num_workers = 0, data_dir = 'chest_xray'):
        
    test_dir = os.path.join(data_dir,'test/')
    # creating test data using Image Folder

    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    test_data  = datasets.ImageFolder(test_dir,transform = data_transform)

    print('Number of test images: ',len(test_data))
    test_loader = torch.utils.data.DataLoader(test_data,batch_size = batch_size,
                                            num_workers = num_workers, shuffle =True)

    return test_loader

def generate_loaders(batch_size  = 32, num_workers = 0, data_dir = 'chest_xray'):
        
    train_dir = os.path.join(data_dir,'train/')
    test_dir = os.path.join(data_dir,'test/')
    # creating train and test data using Image Folder

    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    train_data = datasets.ImageFolder(train_dir,transform=data_transform)
    test_data  = datasets.ImageFolder(test_dir,transform = data_transform)
    print('Number of training images: ', len(train_data))
    print('Number of test images: ',len(test_data))
    train_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,
                                          num_workers = num_workers, shuffle = True)

    test_loader = torch.utils.data.DataLoader(test_data,batch_size = batch_size,
                                            num_workers = num_workers, shuffle =True)


    # classes are folders in each directory with these names
    classes = test_data.classes
    return train_loader,test_loader



if __name__ == "__main__":
    check_gpu()
    generate_loaders()




