from nn_dataset import DEVICE

import os
import copy
import torch
from sklearn.metrics import classification_report, fbeta_score


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, model_name):
    """
    Loop for training a given model.
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    train_losses = []
    val_losses = []

    MODEL_DIR = f'../model/{model_name}/'
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, f'{model_name}.pth')

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            torch.save(best_model_wts, MODEL_PATH)
            print(f"Model checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Stopping early ({patience} epochs without improvement).")
                break
            
    model.load_state_dict(torch.load(MODEL_PATH))  
      
    return model, train_losses, val_losses

def evaluate_model(model, data_loader, name="Test"):
    """Evaluates the model."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    report = classification_report(
        all_targets, 
        all_preds, 
        target_names=['Normal (0)', 'Abnormal (1)'],
        output_dict=True,
        zero_division=0
    )
    
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=['Normal (0)', 'Abnormal (1)'],
        zero_division=0
    ))
    
    abnormal_f1 = report['Abnormal (1)']['f1-score']
    abnormal_recall = report['Abnormal (1)']['recall']
    abnormal_precision = report['Abnormal (1)']['precision']
    fabnormal_f2 = fbeta_score(all_targets, all_preds, beta=2, pos_label=1)

    return abnormal_f1, abnormal_recall, abnormal_precision, fabnormal_f2, all_targets, all_preds
