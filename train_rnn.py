import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from rnn_model import ECG_RNN

AUGMENTED_DIR = "processed_data/augmented"
PREPROCESSED_DIR = "processed_data/preprocessed"
MODEL_DIR = "saved_models"

def load_data():
    """tải dữ liệu và reshape để đưa vào RNN"""
    print("đang tải dữ liệu...")
    X_train = np.load(f"{AUGMENTED_DIR}/X_train_aug.npy")
    y_train = np.load(f"{AUGMENTED_DIR}/y_train_aug.npy")
    
    X_val = np.load(f"{PREPROCESSED_DIR}/X_val.npy")
    y_val = np.load(f"{PREPROCESSED_DIR}/y_val.npy")

    X_train = X_train.reshape(-1, 180, 1)
    X_val = X_val.reshape(-1, 180, 1)

    return X_train, y_train, X_val, y_val

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"đang huấn luyện trên: {device}")

    X_train, y_train, X_val, y_val = load_data()
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = ECG_RNN(input_size=1, hidden_size=32, num_layers=2, model_type='lstm').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 30
    best_val_loss = float('inf')
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("bắt đầu huấn luyện...")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_rnn_model.pth")
            print(f"  -> Đã lưu Checkpoint mới tại epoch {epoch+1}")

    plot_curves(history)

def plot_curves(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rnn_learning_curve.png')
    print("\n huấn luyện hoàn tất. Đã lưu biểu đồ tại 'rnn_learning_curve.png'")

if __name__ == "__main__":
    main()