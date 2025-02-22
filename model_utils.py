from torchvision import models
from torch import optim
from torch import nn
from time import time
import torch
import data_utils

def build_model(arch, learning_rate, num_units):

    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_nodes=25088
    else:
        raise ValueError(f"Model {arch} not supported")
        
    for param in model.parameters():
        param.required_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_nodes, num_units),
                          nn.ReLU(),
                          nn.Dropout(p=0.5),
                          nn.Linear(num_units, 102),
                          nn.LogSoftmax(dim=1))

    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model = build_model(checkpoint['arch'], checkpoint['learning_rate'], checkpoint['num_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def train_model(model, epochs, print_every, optimiser, criterion, train_dataloader, valid_dataloader, device):
    step = 0
    running_loss = 0
    for epoch in range(epochs):
        start_time = time()
        for images, labels in train_dataloader:
            step += 1
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

            if step % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        images, labels = images.to(device), labels.to(device)

                        logps = model(images)
                        valid_loss += criterion(logps, labels).item()

                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                       f"Train loss: {running_loss/(len(train_dataloader)): .5f}.. "
                       f"Validation loss: {valid_loss/len(valid_dataloader): .5f}.. "
                       f"Validation accuracy: {accuracy/len(valid_dataloader): .5f}.. ")
                running_loss = 0
                model.train()
        end_time = time()
        print(f"Elapsed time for epoch {epoch + 1}: {end_time - start_time}")
    print(f"Training and validation complete")
    
def test_model(model, test_dataloader, device):
    model.eval()
    accuracy = 0

    with torch.no_grad():
        for images, labels in test_dataloader:

            images, labels = images.to(device), labels.to(device)

            logps = model.forward(images)

            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    print(f"Testing completed: test accuracy: {accuracy/len(test_dataloader): .5f}.. ")
    model.train()

def predict(image_path, model, topk):
    model.eval()
    with torch.no_grad():
        image = data_utils.transform_image(image_path).unsqueeze(0)
        logps = model(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
        
        for top_class in top_class.numpy()[0]:
            classes.append(class_to_idx_inv[top_class])
        
        return top_p.numpy()[0], classes
