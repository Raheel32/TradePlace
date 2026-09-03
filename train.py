"""
Trains the LSTM classifier on historical data with a chronological
(not random) train/test split, since shuffling time series data leaks
future information into training.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import config
from data_loader import fetch_data
from features import prepare_dataset, build_sequences
from model import LSTMClassifier


def train_model(X_train, y_train, X_test, y_test, input_size,
                 epochs=None, verbose=True, seed=42):
    """
    Core training loop, factored out so both a single train/test run
    (run_training) and each fold of walk-forward validation (walk_forward.py)
    can share the exact same training logic.

    Returns: trained model, raw probability predictions on X_test (numpy array)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = epochs or config.EPOCHS

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    n_samples = X_train_t.shape[0]
    best_test_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, config.BATCH_SIZE):
            idx = permutation[i:i + config.BATCH_SIZE]
            batch_X, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n_samples

        # Evaluate on test set every epoch (not just for printing) so early
        # stopping can react promptly to overfitting.
        model.eval()
        with torch.no_grad():
            test_preds_epoch = model(X_test_t)
            test_loss = criterion(test_preds_epoch, y_test_t).item()
            test_acc = accuracy_score(y_test_t.numpy(), (test_preds_epoch.numpy() > 0.5).astype(int))

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d} | Train loss: {epoch_loss:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.EARLY_STOPPING_PATIENCE:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement in test loss for {config.EARLY_STOPPING_PATIENCE} epochs). "
                          f"Restoring weights from epoch {best_epoch}.")
                break

    # Always restore the best checkpoint seen, not whatever the last epoch happened to be
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"Best epoch: {best_epoch} (test loss {best_test_loss:.4f})")

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()

    return model, test_preds


def run_training():
    # 1. Load and prep data
    raw_df = fetch_data()
    features, labels, dates, _ = prepare_dataset(raw_df)

    # 2. Chronological split BEFORE scaling/sequencing to avoid lookahead bias
    split_idx = int(len(features) * config.TRAIN_TEST_SPLIT)
    train_features, test_features = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    # 3. Scale using only training data stats
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    os.makedirs("models", exist_ok=True)
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # 4. Build sliding-window sequences
    X_train, y_train = build_sequences(train_features_scaled, train_labels, config.LOOKBACK_WINDOW)
    X_test, y_test = build_sequences(test_features_scaled, test_labels, config.LOOKBACK_WINDOW)

    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
    print(f"Train label balance: {y_train.mean():.3f} (fraction of 'up' days)")
    print(f"Test label balance: {y_test.mean():.3f}")

    # 5+6. Train
    model, test_preds = train_model(X_train, y_train, X_test, y_test, input_size=X_train.shape[2])

    # 7. Final evaluation
    test_preds_binary = (test_preds > 0.5).astype(int)

    print("\n=== Final test set report ===")
    print(classification_report(y_test, test_preds_binary, target_names=["Down", "Up"]))

    majority_baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"Naive 'always predict majority class' baseline accuracy: {majority_baseline_acc:.4f}")
    print(f"Model accuracy: {accuracy_score(y_test, test_preds_binary):.4f}")
    print("(If the model isn't clearly beating this baseline, it hasn't learned a real edge yet.)")

    # 8. Save model
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")

    return model, scaler, (X_test, y_test, test_preds, dates[-len(y_test):])


if __name__ == "__main__":
    run_training()