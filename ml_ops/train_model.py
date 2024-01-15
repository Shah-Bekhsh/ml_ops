import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
#from models.model import MyNeuralNet
from models import model

train_data = torch.load('./data/processed/train_data.pt')
train_labels = torch.load('./data/processed/train_labels.pt')
test_data = torch.load('./data/processed/test_images.pt')
test_labels = torch.load('./data/processed/test_target.pt')

batch_size = 64
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

input_size = train_data.size(2) * train_data.size(3)
output_size = 10
hidden_units = 500
model_ = model.MyNeuralNet(input_size, output_size, hidden_units)

print("Our model: \n\n", model_, '\n')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model_.parameters(), lr=0.001)

num_epochs = 10

model_ = model.training(model_ , train_loader, test_loader, criterion, optimizer, num_epochs)

checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'hidden_units': hidden_units,
              'state_dict': model_.state_dict()}

torch.save(checkpoint, './models/checkpoint.pth')

print("Checkpoint: \n\n", checkpoint, '\n')