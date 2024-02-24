import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    newDf = pd.get_dummies(df["home"], drop_first=True)
    newDf["team_wl"] = df["team_wl"].map({"W": 1, "L": 0})
    newDf["score_diff"] = df["team_score"] - df["opp_score"]
    return newDf


def preprocess_and_predict(new_data, model, scaler):
    new_data = preprocess_data(new_data)
    X_new = new_data.drop(["team_wl"], axis=1)
    X_new = scaler.transform(X_new)
    X_new = torch.tensor(X_new, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(X_new)
        predicted = (output.squeeze() > 0.5).float()
    return predicted.numpy()


def main():
    # Load and preprocess data
    df = pd.read_csv("data/nba_gamelog.csv")
    df = preprocess_data(df)

    # Split data
    X = df.drop(["team_wl"], axis=1)
    y = df["team_wl"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # DataLoader
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Model setup
    model = Net(X_train)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()

    # Model evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            predicted = (outputs.squeeze() > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

    print(df.columns)
    # Example prediction
    new_data = pd.DataFrame([])
    predictions = preprocess_and_predict(new_data, model, scaler)
    print(predictions)
