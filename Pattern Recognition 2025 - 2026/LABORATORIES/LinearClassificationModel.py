# -----------------------------------------------------------------------------
# This script file provides fundamental computational functionality for the
# the development of a linear classifier. The learning process will be 
# implemented in the context of a binary classification problem where patterns
# from each class will follow a multi-dimensional gaussian distribution.
# -----------------------------------------------------------------------------

# Import required Python libraries.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# STEP 0: SET RANDOM SEEDS FOR REPRODUCIBILITY
# -----------------------------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)

# Define a function to get the correct training environment for the model.
def get_execution_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS GPU is available!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU is available!")
    else:
        device = torch.device("cpu")
        print("GPU is not available, CPU will be used instead!")
    return device

# -----------------------------------------------------------------------------
# STEP 1: CUSTOM DATASET CLASS DEFINITION FOR SYNTHETIC DATA GENERATION
# -----------------------------------------------------------------------------

class MultivariateGaussianDataset(Dataset):
    def __init__(self, n_samples, mean1, mean2, cov, train_ratio, split, device):
        self.device = device
        self.split = split
        dist1 = torch.distributions.MultivariateNormal(mean1, cov)
        dist2 = torch.distributions.MultivariateNormal(mean2, cov)
        data1 = dist1.sample((n_samples,)).to(self.device)
        data2 = dist2.sample((n_samples,)).to(self.device)
        labels1 = torch.zeros((n_samples, 1), device=self.device)
        labels2 = torch.ones((n_samples, 1), device=self.device)
        self.data = torch.cat((data1, data2), dim=0)
        self.labels = torch.cat((labels1, labels2), dim=0)
        perm = torch.randperm(self.data.size(0), device=self.device)
        self.data, self.labels = self.data[perm], self.labels[perm]
        split_index = int(n_samples * 2 * train_ratio)
        if split == "train":
            self.data = self.data[:split_index]
            self.labels = self.labels[:split_index]
        else:
            self.data = self.data[split_index:]
            self.labels = self.labels[split_index:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# -----------------------------------------------------------------------------
# STEP 2: INSTANTIATE TRAIN AND TESTING DATASET OBJECTS AND DATALOADERS
# -----------------------------------------------------------------------------

device = get_execution_device()

input_dim = 2
n_samples = 500
displacement = 5
cov_scale = 1
train_ratio = 0.8
batch_size = 50

mean1 = torch.randn(input_dim)
mean2 = torch.randn(input_dim) + displacement
cov = torch.eye(input_dim) * cov_scale

train_dataset = MultivariateGaussianDataset(n_samples, mean1, mean2, cov,
                                            train_ratio, split="train",
                                            device=device)
test_dataset = MultivariateGaussianDataset(n_samples, mean1, mean2, cov,
                                           train_ratio, split="test",
                                           device=device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# STEP 3: LINEAR CLASSIFICATION MODEL
# -----------------------------------------------------------------------------

class LinearClassificationModel(nn.Module):
    def __init__(self, input_dim, device):
        super(LinearClassificationModel, self).__init__()
        self.device = device
        self.W1 = nn.Parameter(torch.randn(input_dim).to(self.device))  # Linear weights
        self.W0 = nn.Parameter(torch.tensor(0.0).to(self.device))       # Bias term
    
    def forward(self, x):
        """
        Forward pass for the linear classifier:
        - Computes a linear combination of input features and a bias term.
        - Applies a sigmoid activation to produce probabilities in [0, 1].
        """
        x = x.to(self.device)  # Move input to the specified device
        
        # Linear term: Dot product of inputs with weights
        linear_term = torch.matmul(x, self.W1)
        
        # Unsqueeze bias term to match batch dimensions (adds a new axis)
        bias_term = self.W0.unsqueeze(0)  
        
        # Return sigmoid-activated predictions
        return torch.sigmoid(linear_term + bias_term)

# -----------------------------------------------------------------------------
# STEP 4: MODEL TRAINING AND EVALUATION METHODS
# -----------------------------------------------------------------------------

def train_model(model, train_loader, test_loader, criterion, optimizer, device):
    previous_epoch_loss = float('inf')  
    epoch = 0
    while True:
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 1), labels)  # Ensure consistent shapes
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            total_predictions += labels.size(0)
            predicted = outputs.round()
            correct_predictions += (predicted.view(-1, 1) == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        test_loss, test_accuracy = evaluate_accuracy(model, test_loader, criterion)
        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.8f}, "
              f"Training Accuracy: {epoch_accuracy:.2f}, "
              f"Testing Accuracy: {test_accuracy:.2f}")
        
        if epoch_loss > previous_epoch_loss:
            print("Stopping training as the loss increased from the previous epoch.")
            break
            
        previous_epoch_loss = epoch_loss
        epoch += 1

def evaluate_accuracy(model, data_loader, criterion):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 1), labels)  # Ensure consistent shapes
            running_loss += loss.item()
            total_predictions += labels.size(0)
            predicted = outputs.round()
            correct_predictions += (predicted.view(-1, 1) == labels).sum().item()
    loss = running_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    return loss, accuracy

# -----------------------------------------------------------------------------
# STEP 5: MODEL INITIALIZATION, TRAINING AND EVALUATION
# -----------------------------------------------------------------------------

model = LinearClassificationModel(input_dim, device)
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_model(model, train_loader, test_loader, criterion, optimizer, device)

# -----------------------------------------------------------------------------
# STEP 6: MODEL VISUALIZATION
# -----------------------------------------------------------------------------

def group_data(data_loader):
    class1_patterns = []
    class2_patterns = []
    for inputs, labels in data_loader:
        class1_patterns.append(inputs[labels.squeeze() == 0])
        class2_patterns.append(inputs[labels.squeeze() == 1])
    class1_patterns = torch.cat(class1_patterns, dim=0).cpu()
    class2_patterns = torch.cat(class2_patterns, dim=0).cpu()
    return class1_patterns, class2_patterns

def plot_data(model, train_loader, test_loader, device):
    # Ensure the model only works for 2D input
    if train_loader.dataset.data.size(1) != 2:
        print("Visualization is only supported for 2D data.")
        return
    
    train_class1, train_class2 = group_data(train_loader)
    test_class1, test_class2 = group_data(test_loader)

    # Determine dynamic plot limits based on data
    all_data = torch.cat([train_class1, train_class2, test_class1, test_class2], dim=0)
    x_min, x_max = all_data[:, 0].min() - 1, all_data[:, 0].max() + 1
    y_min, y_max = all_data[:, 1].min() - 1, all_data[:, 1].max() + 1

    # Create a grid over the data range
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

    # Predict over the grid to plot the decision boundary
    with torch.no_grad():
        predictions = model(grid_points).reshape(xx.shape).cpu().numpy()

    # Plot the training and testing data
    plt.figure(figsize=(8, 6))
    plt.scatter(train_class1[:, 0], train_class1[:, 1], c='blue', label='Class A (Train)', alpha=0.6)
    plt.scatter(train_class2[:, 0], train_class2[:, 1], c='red', label='Class B (Train)', alpha=0.6)
    plt.scatter(test_class1[:, 0], test_class1[:, 1], c='cyan', label='Class A (Test)', alpha=0.9)
    plt.scatter(test_class2[:, 0], test_class2[:, 1], c='orange', label='Class B (Test)', alpha=0.9)

    # Plot the decision boundary as a contour
    plt.contour(xx, yy, predictions, levels=[0.5], colors='black', linewidths=1, linestyles='--')

    # Add legend, title, and axis labels
    plt.legend()
    plt.title("Data with Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

plot_data(model, train_loader, test_loader, device)