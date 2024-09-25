import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from models.neural_network import NeuralNetwork

# Step 1: Load and preprocess the Iris dataset
iris = fetch_openml(name='iris', version=1)
X, y = iris.data, iris.target

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

model = NeuralNetwork()

# Train the model and visualize the loss surface
model.train_with_loss_surface(X_train, y_train, num_epochs=100)