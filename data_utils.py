from PIL import Image
from torch.utils.data import DataLoader as dl
import torchvision.transforms as trans
import torchvision.datasets as ds

def load_data(data_dir):
    print(f"Loading data")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = trans.Compose([
                                    trans.RandomResizedCrop(244), 
                                    trans.RandomRotation(15), 
                                    trans.RandomHorizontalFlip(),
                                    trans.ToTensor(),
                                    trans.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                                    ])
    test_valid_transform = trans.Compose([
                                        trans.Resize(225), 
                                        trans.CenterCrop(244), 
                                        trans.ToTensor(),
                                        trans.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                        ])

    train_dataset = ds.ImageFolder(train_dir, transform=train_transform)
    test_dataset = ds.ImageFolder(test_dir, transform=test_valid_transform)
    valid_dataset = ds.ImageFolder(valid_dir, transform=test_valid_transform)


    train_dataloader = dl(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = dl(test_dataset, batch_size=64, shuffle=True)
    valid_dataloader = dl(valid_dataset, batch_size=64, shuffle=True)
    
    return train_dataset, train_dataloader, test_dataloader, valid_dataloader

def transform_image(image):
    image = Image.open(image)
    transformed_image = trans.Compose([trans.Resize(255),
                                                  trans.CenterCrop(224),
                                                  trans.ToTensor(),
                                                  trans.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])
                                         ])
    
    return transformed_image(image)
