import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1  = nn.Linear(4, 32)
        self.fc2  = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    
def makeModel():
    # Create the model
    model = RegressionModel()
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Define the loss function
    criterion = nn.MSELoss()

    return model, optimizer, criterion

def trainModel(epochs, train_loader, val_loader):
    
    model, optimizer, criterion = makeModel()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item() * val_inputs.size(0)
            
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss}, Val Loss: {val_loss}")

    return model