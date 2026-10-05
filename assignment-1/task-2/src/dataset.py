import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class AutoformerDataset(Dataset):
    """
    PyTorch Dataset for the Leaderboard Challenge.

    Key improvements over the original:
      • Target and covariates are loaded separately so the full 43,824-row
        covariate file is available (including the 168 future-horizon rows).
      • Returns future covariates for the decoder alongside the encoder input.
      • Global normalization is fitted on training data only.
    """
    def __init__(self, target_path, cov_path=None, cov_cols=None,
                 context_len=96, pred_len=168, mode='train', val_split=0.8):
        self.context_len = context_len
        self.pred_len = pred_len
        self.mode = mode

        # ---- 1. Load target ----
        df_target = pd.read_csv(target_path)
        target = df_target['value'].values.astype(np.float32)
        n_target = len(target)  # 43,656 rows

        # ---- 2. Load covariates (may span beyond target into the test horizon) ----
        self.has_cov = bool(cov_path and cov_cols)
        if self.has_cov:
            df_cov = pd.read_csv(cov_path)
            if cov_cols == 'all':
                feature_cols = [c for c in df_cov.columns if c != 'time_idx']
            else:
                feature_cols = list(cov_cols)
            self.cov_data = df_cov[feature_cols].values.astype(np.float32)
            self.num_cov = self.cov_data.shape[1]
        else:
            self.cov_data = None
            self.num_cov = 0

        # ---- 3. Chronological train/val split ----
        train_end = int(n_target * val_split)

        # ---- 4. Fit normalization on covariates only ----
        # The target is NOT globally normalized here because the Autoformer
        # uses RevIN (Reversible Instance Normalization) internally to handle
        # heavy-tailed, non-stationary level shifts on a per-window basis.
        self.target_norm = target

        if self.has_cov:
            self.cov_mean = self.cov_data[:train_end].mean(axis=0)
            self.cov_std = self.cov_data[:train_end].std(axis=0)
            self.cov_std[self.cov_std == 0] = 1.0
            
            # Keep binary features as 0/1 by forcing mean=0 and std=1
            binary_features = {'feature_G', 'feature_H', 'feature_I', 'feature_J'}
            for i, col_name in enumerate(feature_cols):
                if col_name in binary_features:
                    self.cov_mean[i] = 0.0
                    self.cov_std[i] = 1.0
                    
            self.cov_norm = (self.cov_data - self.cov_mean) / self.cov_std
        else:
            self.cov_norm = None

        # ---- 5. Set index ranges for each mode ----
        if mode == 'train':
            self.offset = 0
            self.length = train_end - context_len - pred_len + 1
        elif mode == 'val':
            self.offset = train_end - context_len
            self.length = n_target - self.offset - context_len - pred_len + 1
        elif mode == 'test':
            self.offset = n_target - context_len
            self.length = 1

        self.n_target = n_target

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        s_begin = self.offset + index
        s_end = s_begin + self.context_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # ---- Encoder input: [context_len, 1 + num_cov] ----
        target_ctx = self.target_norm[s_begin:s_end].reshape(-1, 1)
        if self.has_cov:
            cov_ctx = self.cov_norm[s_begin:s_end]
            x_enc = np.concatenate([target_ctx, cov_ctx], axis=1)
        else:
            x_enc = target_ctx

        # ---- Future covariates: [pred_len, num_cov] ----
        if self.has_cov and r_end <= len(self.cov_norm):
            x_dec_cov = self.cov_norm[r_begin:r_end]
        elif self.has_cov:
            # Fallback (should not happen with the provided data)
            x_dec_cov = np.zeros((self.pred_len, self.num_cov), dtype=np.float32)
        else:
            # No covariates: provide a dummy that the model will ignore
            x_dec_cov = np.zeros((self.pred_len, 1), dtype=np.float32)

        if self.mode == 'test':
            return torch.tensor(x_enc), torch.tensor(x_dec_cov)

        # ---- Target horizon: [pred_len] ----
        y = self.target_norm[r_begin:r_end]
        return torch.tensor(x_enc), torch.tensor(x_dec_cov), torch.tensor(y)

    def inverse_transform_target(self, data):
        """
        Predictions are already in physical units because RevIN denormalizes
        them inside the model. No global inverse transform is needed.
        """
        return data
