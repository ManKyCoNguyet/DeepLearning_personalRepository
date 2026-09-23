"""
PyTorch MLP interface.
MODEL_06 = MLP PyTorch
Class: HousePriceMLP(input_dim, hidden_dims=(256,128,64), dropout=0.20)
Functions:
- train_mlp(...) -> model, history, training_time_seconds, device
- predict_mlp(model, X, device=None) -> numpy.ndarray

TODO TV3: MLP PyTorch, training/validation loop, early stopping,
best checkpoint, history and W&B tracking.
"""
import csv
import time
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import wandb

class HousePriceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.20):
        super().__init__()
        # Code mẫu; TV3 hoàn thiện/kiểm tra theo contract.
        layers = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
            if i < len(hidden_dims)-1:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_mlp(X_train, y_train, X_val, y_val, epochs=100, batch_size=64,
              lr=1e-3, patience=15, seed=42, dropout=0.20, save_checkpoint=False):

    
    # TODO TV3: # MLP training with early stopping, best checkpoint and W&B tracking.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    wandb.init(
        project="lab02-house-price",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "hidden_dims": (256, 128, 64),
            "dropout": dropout,
            "optimizer": "Adam",
            "loss": "MSELoss"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HousePriceMLP(
        input_dim=X_train.shape[1],
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Log-transform target
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_log, dtype=torch.float32).reshape(-1, 1)
    )

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(
        y_val_log,
        dtype=torch.float32,
        device=device
    ).reshape(-1, 1)

    history = {"train_loss": [], "val_loss": []}
    start = time.perf_counter()

    best_val_loss = float("inf")

    # Đường dẫn cố định về thư mục experiments của project
    project_root = Path(__file__).resolve().parent.parent
    experiments_dir = project_root / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = experiments_dir / "best_mlp.pt"
    history_path = experiments_dir / "mlp_history.csv"

    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_ds)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # # Early stopping and best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            if save_checkpoint:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    # Load lại mô hình tốt nhất 
    if save_checkpoint and checkpoint_path.exists():
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )

    if save_checkpoint:
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

            for epoch, (train_loss, val_loss) in enumerate(
                zip(history["train_loss"], history["val_loss"]), start=1
            ):
                writer.writerow([epoch, train_loss, val_loss])
       
    training_time_seconds = time.perf_counter() - start

    wandb.finish()

    return model, history, training_time_seconds, device

def predict_mlp(model, X, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        pred_log = model(x).squeeze(1).detach().cpu().numpy()

        # Đưa prediction từ log(SalePrice) về SalePrice gốc
        return np.expm1(pred_log)