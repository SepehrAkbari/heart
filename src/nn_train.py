from setup import *
from nn_dataset import *
from nn_models import *
from nn_traintest_loop import train_model

import torch.nn as nn
import torch.optim as optim
import numpy as np

# Training BaselineFNN
model = BaselineFNN(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "BaselineFNN"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
num_epochs = 50

model_BaselineFNN, train_loss_BaselineFNN, val_loss_BaselineFNN = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_BaselineFNN, val_loss=val_loss_BaselineFNN)


# Training EnhancedFNN
model = EnhancedFNN(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "EnhancedFNN"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
num_epochs = 50

model_EnhancedFNN, train_loss_EnhancedFNN, val_loss_EnhancedFNN = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_EnhancedFNN, val_loss=val_loss_EnhancedFNN)


# Training BaselineCNN
model = BaselineCNN(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "BaselineCNN"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
num_epochs = 50

model_BaselineCNN, train_loss_BaselineCNN, val_loss_BaselineCNN = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_BaselineCNN, val_loss=val_loss_BaselineCNN)


# Training MediumCNN
model = MediumCNN(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "MediumCNN"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
num_epochs = 50

model_MediumCNN, train_loss_MediumCNN, val_loss_MediumCNN = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_MediumCNN, val_loss=val_loss_MediumCNN)


# Training EnhancedCNN
model = EnhancedCNN(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "EnhancedCNN"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
num_epochs = 50

model_EnhancedCNN, train_loss_EnhancedCNN, val_loss_EnhancedCNN = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_EnhancedCNN, val_loss=val_loss_EnhancedCNN)


# Training seCNN
model = seCNN(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "seCNN"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
num_epochs = 50

model_seCNN, train_loss_seCNN, val_loss_seCNN = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_seCNN, val_loss=val_loss_seCNN)


# Training DFFN
model = DFFN(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "DFFN"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
num_epochs = 50

model_DFFN, train_loss_DFFN, val_loss_DFFN = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_DFFN, val_loss=val_loss_DFFN)


# Training AugmentedCNN
model = AugmentedCNN(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "AugmentedCNN"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
num_epochs = 50

model_AugmentedCNN, train_loss_AugmentedCNN, val_loss_AugmentedCNN = train_model(model=model, train_loader=train_loader_aug, val_loader=val_loader_aug, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_AugmentedCNN, val_loss=val_loss_AugmentedCNN)


# Training BaselineResnetTransfer
model = BaselineResnetTransfer(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "BaselineResnetTransfer"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
patience = 50
num_epochs = 50

model_BaselineResnetTransfer, train_loss_BaselineResnetTransfer, val_loss_BaselineResnetTransfer = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_BaselineResnetTransfer, val_loss=val_loss_BaselineResnetTransfer)


# Training EnhancedResnetTransfer
model = EnhancedResnetTransfer(INPUT_SIZE, NUM_CLASSES)
model.to(DEVICE)

model_name = "EnhancedResnetTransfer"
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.005)
patience = 10
num_epochs = 50

model_EnhancedResnetTransfer, train_loss_EnhancedResnetTransfer, val_loss_EnhancedResnetTransfer = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=num_epochs, patience=patience, model_name=model_name)

np.savez(f'../model/{model_name}/{model_name}_losses.npz', train_loss=train_loss_EnhancedResnetTransfer, val_loss=val_loss_EnhancedResnetTransfer)