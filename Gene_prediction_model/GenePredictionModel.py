import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import pandas as pd


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Data loading and splitting
def data_load():
    data = np.load('1000G_reqnorm_float64.npy')

    num_genes, num_samples = data.shape
    print(f"Number of genes (rows): {num_genes}")
    print(f"Number of samples (columns): {num_samples}")

    #Z-score normalization
    data_means = data.mean(axis=1, keepdims=True)
    data_stds = data.std(axis=1, keepdims=True) + 1e-3
    data = (data - data_means) / data_stds

    num_lm = 943
    X = data[:num_lm, :].T  # (samples × genes_lm)
    Y = data[num_lm:, :].T  # (samples × genes_other)

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    #Splitting of X & y
    X = X[indices]
    Y = Y[indices]
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    #defining train and test size
    train_end = int(0.7 * num_samples)
    val_end = int(0.9 * num_samples)

    #Train,val and test split
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    def to_ds(x, y):
        return TensorDataset(torch.tensor(x, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.float32))
    return to_ds(X_train, Y_train), to_ds(X_val, Y_val), to_ds(X_test,Y_test), input_dim, output_dim


# Defining Model Architecture
class GenePredictionFFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=800, hidden2=750, hidden3=500, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim)
        )

#Forward pass
    def forward(self, x):
        return self.layers(x)

# Training & Testing
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)

def test(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)




#Main function
def main():
    train_ds, val_ds, test_ds, input_dim, output_dim=data_load()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32,drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=32,drop_last=True)

#Hyperparameters
    param_grid = {
        "hidden1": [800, 700],
        "hidden2": [600,500],
        "hidden3": [500,250],
        "dropout": [0.2, 0.3],
        "lr": [1e-3, 5e-4]
    }

#Defining loss function
    loss_fn = nn.MSELoss()

#Hyperparameter tuning
    best_params, best_val_loss = None, float("inf")
    for h1 in param_grid["hidden1"]:
        for h2 in param_grid["hidden2"]:
            for h3 in param_grid["hidden3"]:
                for dr in param_grid["dropout"]:
                    for lr in param_grid["lr"]:
                        model = GenePredictionFFN(input_dim,output_dim, h1, h2,h3, dr).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                        # Short training for tuning
                        for epoch in range(50):
                            train(train_loader, model, loss_fn, optimizer)
                        val_loss = test(val_loader, model, loss_fn)
                        print(f"h1={h1}, h2={h2}, h3={h3}, dr={dr}, lr={lr} | Val Loss={val_loss:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = (h1, h2,h3, dr, lr)

    print(f"\nBest Params: {best_params} | Best Val Loss: {best_val_loss:.4f}")

    # Final training using train + val sets
    train_val_loader = DataLoader(ConcatDataset([train_ds, val_ds]), batch_size=32, shuffle=True,drop_last=True)

    # Best Params: (700, 600, 500, 0.2, 0.001)
    h1, h2, h3, dr, lr = best_params
    model = GenePredictionFFN(input_dim, output_dim, h1, h2, h3, dr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_epochs = 1024
    train_losses, test_losses = [], []

    for t in range(n_epochs):
        train_loss = train(train_val_loader, model, loss_fn, optimizer)
        test_loss = test(val_loader, model, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if (t + 1) % 50 == 0:
            print(f"Epoch {t + 1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # Plot training curve (for train and test loss)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    #Saving the trained model
    torch.save(model.state_dict(), "tg_prediction_from_lm_genes.pth")
    print("Saved PyTorch Model State to tg_prediction_from_lm_genes.pth")

    # Final Test Evaluation
    # Load model from saved file
    final_model = GenePredictionFFN(input_dim, output_dim, h1, h2, h3, dr).to(device)
    final_model.load_state_dict(torch.load("tg_prediction_from_lm_genes.pth"))
    final_model.eval()

    # Evaluate on final test set
    final_test_loss = test(test_loader, final_model, loss_fn)
    print(f"\nFinal Test Loss (unseen data): {final_test_loss:.4f}")

    # np.save('1000G_X_train.npy', X_train)
    # np.save('1000G_Y_train.npy', Y_train)
    #
    # np.save('1000G_X_val.npy', X_val)
    # np.save('1000G_Y_val.npy', Y_val)
    #
    # np.save('1000G_X_test.npy', X_test)
    # np.save('1000G_Y_test.npy', Y_test)



if __name__ == '__main__':
    main()
