import pandas as pd
from mlxtend.preprocessing import minmax_scaling
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger les données
data = pd.read_csv('data (1).csv')
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
columns_to_scale = data.columns[1:]
data_scaled = minmax_scaling(data, columns=columns_to_scale)
X = data_scaled
y = data['diagnosis']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values)

train_set = TensorDataset(x_train_tensor, y_train_tensor)
batch_size = 50
train_set_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = TensorDataset(torch.FloatTensor(X_test.values), torch.FloatTensor(y_test.values))
test_set_loader = DataLoader(test_set, shuffle=False)

# Définir le modèle de réseau neuronal
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

input_size = x_train_tensor.shape[1]
model = NeuralNetwork(input_size)
criterion = nn.BCELoss()
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Fonction d'entraînement
def train(model, criterion, optimizer, dataloader):
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    vp, fp, fn, vn = 0, 0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        predictions = (outputs.squeeze() > 0.5).float()
        correct_predictions += (predictions == labels).sum().item()
        vp += ((predictions == 1) & (labels == 1)).sum().item()
        fp += ((predictions == 1) & (labels == 0)).sum().item()
        fn += ((predictions == 0) & (labels == 1)).sum().item()
        vn += ((predictions == 0) & (labels == 0)).sum().item()
        total_samples += labels.size(0)

    average_loss = epoch_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    precision = vp / (vp + fp) if (vp + fp) != 0 else 0.0

    print(f'Epoch Loss: {average_loss:.4f}, Epoch Accuracy: {accuracy:.4f}, Epoch Precision: {precision:.4f}')
    print(f'TP: {vp}, FP: {fp}, FN: {fn}, VN: {vn}')

    return average_loss, accuracy, precision

# Initialiser des listes pour stocker les valeurs
train_losses = []
train_accuracies = []
train_precisions = []

num_epochs = 25


for epoch in range(num_epochs):
    train_loss, train_accuracy, train_precision = train(model, criterion, optimizer, train_set_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)

# Fonction de test
def test(model, criterion, dataloader):
    test_losses = []
    test_accuracies = []
    test_precisions = []

    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    true_positives = 0
    false_positives = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(dim=1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()

        # Calcul des performances
        average_loss = test_loss / len(dataloader)
        accuracy = correct_predictions / len(dataloader.dataset)
        precision = true_positives / (true_positives + false_positives)

        test_losses.append(average_loss)
        test_accuracies.append(accuracy)
        test_precisions.append(precision)

    return average_loss, accuracy, precision, test_losses, test_accuracies, test_precisions

test_loss, test_accuracy, test_precision, test_losses, test_accuracies, test_precisions = test(model, criterion, test_set_loader)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Precision: {test_precision * 100:.2f}%")

print("Liste des losses de test:", test_losses)
print("Liste des accuracies de test:", test_accuracies)
print("Liste des precisions de test:", test_precisions)

epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 4))

#loss
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss en fonction Epochs')
plt.legend()

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy en fonction  Epochs')
plt.legend()

# Precision
plt.subplot(1, 3, 3)
plt.plot(epochs, train_precisions, label='Training Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Training Precision en fonction Epochs')
plt.legend()

plt.tight_layout()
plt.show()
