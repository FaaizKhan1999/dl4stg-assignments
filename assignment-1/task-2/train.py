import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from src.dataset import AutoformerDataset
from src.autoformer import Autoformer
from src.engine import train_one_epoch, evaluate, generate_leaderboard_forecast

COV_CONFIGS = {
    'none': [],
    'binary': ['feature_G', 'feature_H', 'feature_I', 'feature_J'],
    'feature_d': ['feature_D'],
    'binary_and_d': ['feature_D', 'feature_G', 'feature_H', 'feature_I', 'feature_J'],
    'all': 'all'
}


def seed_everything(seed):
    """Locks all random number generators for reproducible multi-seed ablations."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cov_cols = COV_CONFIGS[args.cov_config]
    print(f"Using device: {device} | Seed: {args.seed} | Covariates: {args.cov_config}")

    # ---- 1. Dataset & DataLoader ----
    cov_path = "/kaggle/input/datasets/focusedkognition/dl4stg/data/optional_external_data.csv" if cov_cols else None

    train_dataset = AutoformerDataset(
        "/kaggle/input/datasets/focusedkognition/dl4stg/data/student_train.csv", cov_path, cov_cols,
        context_len=args.context_len, pred_len=168, mode='train'
    )
    val_dataset = AutoformerDataset(
        "/kaggle/input/datasets/focusedkognition/dl4stg/data/student_train.csv", cov_path, cov_cols,
        context_len=args.context_len, pred_len=168, mode='val'
    )
    test_dataset = AutoformerDataset(
        "/kaggle/input/datasets/focusedkognition/dl4stg/data/student_train.csv", cov_path, cov_cols,
        context_len=args.context_len, pred_len=168, mode='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ---- 2. Model ----
    num_cov = 10 if cov_cols == 'all' else len(cov_cols) if cov_cols else 0
    model = Autoformer(
        context=args.context_len,
        horizon=168,
        label_len=args.label_len,
        in_channels=1 + num_cov,
        d=args.d_model,
        heads=args.n_heads,
        enc_layers=args.e_layers,
        dec_layers=args.d_layers,
        dropout=args.dropout,
        kernel=args.kernel,
        c_out=1,
        num_future_cov=num_cov,
    ).to(device)

    print(f"Total Trainable Parameters (P): {model.parameter_count}")

    # ---- 3. Optimizer, Scheduler & Loss ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Warmup for the first 3 epochs, then cosine annealing
    warmup_epochs = min(3, args.epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=args.learning_rate * 0.01
    )
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    # L1 (MAE) is more robust for heavy-tailed targets
    criterion = nn.MSELoss()

    # ---- 4. Training Loop with Early Stopping ----
    best_val_rmse = float('inf')
    best_metrics = {}
    patience_counter = 0
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./results/", exist_ok=True)
    checkpoint_path = f"./checkpoints/autoformer_cov_{args.cov_config}_seed_{args.seed}.pth"
    submission_path = f"./results/submission_{args.cov_config}_seed_{args.seed}.txt"
    metrics_path = f"./results/metrics_{args.cov_config}_seed_{args.seed}.json"

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            max_grad_norm=args.max_grad_norm
        )
        val_loss, val_mae, val_mse, val_rmse, val_smape = evaluate(
            model, val_loader, criterion, device, val_dataset
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | "
            f"Val sMAPE: {val_smape:.4f} | LR: {current_lr:.6f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_metrics = {
                'rmse': float(val_rmse),
                'mae': float(val_mae),
                'smape': float(val_smape),
                'epoch': epoch + 1
            }
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Best model saved! (Val RMSE: {best_val_rmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # ---- 5. Load best & generate submission ----
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    generate_leaderboard_forecast(
        model, test_loader, device, test_dataset, save_path=submission_path
    )
    
    import json
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 Autoformer Training")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cov_config', type=str, default='none', choices=COV_CONFIGS.keys())
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Peak learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--label_len', type=int, default=48, help='Decoder label length (warm-start from context)')
    parser.add_argument('--context_len', type=int, default=96, help='Encoder context length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--kernel', type=int, default=25, help='Moving average kernel size (must be odd)')
    parser.add_argument('--max_grad_norm', type=float, default=100.0, help='Max gradient norm for clipping')

    args = parser.parse_args()
    main(args)
