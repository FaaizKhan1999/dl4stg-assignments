import torch
import numpy as np
from tqdm import tqdm
from src.metrics import calculate_metrics


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    Training loop for one epoch with gradient clipping.
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_x, batch_cov, batch_y in progress_bar:
        batch_x = batch_x.float().to(device)
        batch_cov = batch_cov.float().to(device)
        batch_y = batch_y.float().to(device)

        optimizer.zero_grad()

        # Forward pass (encoder input + future covariates)
        outputs = model(batch_x, batch_cov)

        # Loss on normalized data
        mean = model.revin.mean.squeeze(-1)
        stdev = model.revin.stdev.squeeze(-1)
        norm_outputs = (outputs - mean) / stdev
        norm_y = (batch_y - mean) / stdev
        loss = criterion(norm_outputs, norm_y)

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, dataset):
    """
    Validation loop that scales predictions back to physical units
    before calculating MAE, sMAPE, and RMSE.
    """
    model.eval()
    total_loss = 0

    preds = []
    trues = []

    with torch.no_grad():
        for batch_x, batch_cov, batch_y in dataloader:
            batch_x = batch_x.float().to(device)
            batch_cov = batch_cov.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x, batch_cov)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Inverse transform to calculate metrics in original scale
    preds_inv = dataset.inverse_transform_target(preds)
    trues_inv = dataset.inverse_transform_target(trues)

    mae, mse, rmse, smape = calculate_metrics(preds_inv, trues_inv)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, mae, mse, rmse, smape


def generate_leaderboard_forecast(model, test_loader, device, dataset,
                                   save_path="results/submission.txt"):
    """
    Generates the final 168-step forecast on the unseen test window,
    transforms it back to physical units, and formats it as a
    comma-separated string for the leaderboard.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch[0].float().to(device)
            batch_cov = batch[1].float().to(device)
            outputs = model(batch_x, batch_cov)
            preds.append(outputs.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    preds_inv = dataset.inverse_transform_target(preds)

    # The output is a single 168-step horizon
    final_forecast = preds_inv[0]

    # Format as comma-separated values
    submission_string = ", ".join([f"{val:.4f}" for val in final_forecast])

    # Save to file to avoid manual transcription errors
    with open(save_path, "w") as f:
        f.write(submission_string)

    print(f"Submission string saved to {save_path}")
    print(f"Total Parameters (P): {model.parameter_count}")
    return final_forecast
