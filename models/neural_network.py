import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = self.relu(self.hidden(x))
        return self.output(hidden)

class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size=64, output_size=2, learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = NeuralNetwork(input_size, hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def fit(self, X, y):
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        for epoch in range(self.epochs):
            self.model.train()
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self
    
    def predict(self, X):
        self.model.eval()
        X = torch.FloatTensor(X)
        with torch.no_grad():
            y_pred = self.model(X)
        return torch.argmax(y_pred, dim=1).numpy()
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


ibd_data = pd.read_csv("./data/hmp/normalized_hmp_final.csv", index_col=0)
ibd_data["Label"] = 1
ibd_data.iloc[:, :-1] = ibd_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
hmp_data = pd.read_csv("./data/ibdmdb/normalized_ibd_final.csv", index_col=0)
hmp_data["Label"] = 0
ibd_data.iloc[:, :-1] = ibd_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

combined_data = pd.concat([ibd_data, hmp_data], axis=0).dropna()

print(combined_data)

# combine the datasets
X = torch.tensor(combined_data.drop("Label", axis=1).values, dtype=torch.float32)
y = torch.tensor(combined_data["Label"].values, dtype=torch.int64)


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

# create the model
model = TorchClassifier(X.shape[1], hidden_size=64, output_size=2, epochs=1000)
scores = cross_val_score(model, X, y, cv=10)
print(scores)
print("Mean: ",scores.mean())

# Ensure both labels are passed even if only one is present in the prediction
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Create display with both labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "IBD"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Analyze first layer weights
weights = model.hidden.weight.detach().numpy()
feature_magnitudes = abs(weights).mean(axis=0)

# Plot feature magnitudes
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_magnitudes)), feature_magnitudes)
plt.title("Feature Magnitudes from First Layer Weights")
plt.xlabel("Features")
plt.ylabel("Average Magnitude")
plt.show()

# Analyze first layer weights
weights = model.output.weight.detach().numpy()
feature_magnitudes = abs(weights).mean(axis=0)

# Plot feature magnitudes
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_magnitudes)), feature_magnitudes)
plt.title("Feature Magnitudes from First Layer Weights")
plt.xlabel("Features")
plt.ylabel("Average Magnitude")
plt.show()

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

