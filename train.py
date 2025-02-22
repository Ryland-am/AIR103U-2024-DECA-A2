import argparse
import data_utils
import model_utils
import json
import torch.nn as nn
import torch.optim as optim
import torch

from torch.utils.data import DataLoader as dl

parser = argparse.ArgumentParser()

parser.add_argument('data_dir')
parser.add_argument('--save_dir', type = str, default = '')
parser.add_argument('--arch', type = str, default = 'vgg13')
parser.add_argument('--learning_rate', type = float, default = 0.01)
parser.add_argument('--num_units', type = int, default = 512)
parser.add_argument('--epochs', type = int, default = 5)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

num_units = args.num_units
learning_rate = args.learning_rate
epochs = args.epochs
arch = args.arch
save_dir = args.save_dir
data_dir = args.data_dir
save_path = save_dir + 'checkpoint.pth'

print(f"Using device: {device}")

# --------------------------Loading data----------------------
print(f"Loading data")
train_dataset, train_dataloader, test_dataloader, valid_dataloader = data_utils.load_data(data_dir)
print(f"Data loading complete")

#------------------------Label Mapping-------------------------
print(f"Mapping labels")
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(f"Label mapping complete")

#------------------------Build Model--------------------------
print(f"Building model with arguments: Architecture: {arch}, Numer of units: {num_units}, Learning rate: {learning_rate}")
model = model_utils.build_model(arch, learning_rate, num_units)
criterion = nn.NLLLoss()
optimiser = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)

print(f"Model build completed")

#------------------------------Train Model--------------------------
print(f"Training model for {epochs} epochs")
print_every = 50
model_utils.train_model(model, epochs, print_every, optimiser, criterion, train_dataloader, valid_dataloader, device)
          
#---------------------------Test Model---------------------------
print(f"Testing model")
model_utils.test_model(model, test_dataloader, device)
print(f"Testing done")

#------------------------------Save checkpoint----------------------
print(f"Saving checkpoint")
model.class_to_isx = train_dataset.class_to_idx
checkpoint = {
    'epochs': epochs,
    'learning_rate': learning_rate,
    'state_dict': model.state_dict(),
    'classifier': model.classifier,
    'class_to_idx': train_dataset.class_to_idx,
    'arch': arch,
    'num_units': num_units
}
save_path = save_dir + 'checkpoint.pth'
torch.save(checkpoint, save_path)
print(f"Model saved to {save_path}")
