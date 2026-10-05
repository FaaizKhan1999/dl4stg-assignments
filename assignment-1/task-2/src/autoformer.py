import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., 2022).
    Normalizes each window independently and restores statistics on the output,
    which helps handle non-stationary series with level shifts.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        """
        x: [B, L, C]  (norm) or [B, H, C] (denorm)
        mode: 'norm' to normalize, 'denorm' to restore.
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            # CRITICAL FIX: Variance + eps prevents NaNs and noise explosions on flat windows
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
            return x


class SeriesDecomposition(nn.Module):
    """
    Splits the input into a slow trend and a repeating remainder stream
    via a centered moving-average filter with replicate boundary padding.
    Adapted from Task 1.
    """
    def __init__(self, kernel):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x):
        padding = self.kernel // 2

        x_transposed = x.transpose(1, 2)
        x_padded = F.pad(x_transposed, (padding, padding), mode='replicate')
        trend_transposed = F.avg_pool1d(x_padded, self.kernel, stride=1)
        trend = trend_transposed.transpose(1, 2)

        remainder = x - trend
        return remainder, trend


def delay_scores(queries, keys):
    """
    Scores temporal delays efficiently in O(L log L) using FFT.
    Supplied from Task 1.
    """
    queries = queries - queries.mean(-1, keepdim=True)
    keys = keys - keys.mean(-1, keepdim=True)
    spectrum = torch.fft.rfft(queries, dim=-1) * torch.fft.rfft(keys, dim=-1).conj()
    return torch.fft.irfft(spectrum, n=queries.shape[-1], dim=-1).mean(2)


def aggregate_delays(values, delays, weights, L_q):
    B, heads, features, L_kv = values.shape
    K = delays.shape[-1]

    t = torch.arange(L_q, device=values.device).view(1, 1, 1, L_q)
    delays_expanded = delays.unsqueeze(-1)
    
    # CRITICAL FIX: Modulo by the original unpadded encoder length (L_kv)
    indices = (t - delays_expanded) % L_kv

    indices = indices.unsqueeze(3).expand(B, heads, K, features, L_q)
    values_expanded = values.unsqueeze(2).expand(B, heads, K, features, L_kv)
    gathered_values = torch.gather(values_expanded, dim=-1, index=indices)

    weights_expanded = weights.unsqueeze(3).unsqueeze(4)
    return (gathered_values * weights_expanded).sum(dim=2)


class AutoCorrelationMixer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, cross=None):
        B, L_q, D = x.shape
        source = cross if cross is not None else x
        L_kv = source.shape[1]

        q = self.query(x).view(B, L_q, self.n_heads, self.d_head).permute(0, 2, 3, 1)
        k = self.key(source).view(B, L_kv, self.n_heads, self.d_head).permute(0, 2, 3, 1)
        v = self.value(source).view(B, L_kv, self.n_heads, self.d_head).permute(0, 2, 3, 1)

        # Save original v for unpadded cyclic rolling
        v_original = v

        L_max = max(L_q, L_kv)
        if L_q < L_max:
            q = F.pad(q, (0, L_max - L_q))
        if L_kv < L_max:
            k = F.pad(k, (0, L_max - L_kv))
            # DO NOT pad v! FFT only needs q and k padded for delay_scores

        scores = delay_scores(q, k)
        K_top = min(math.ceil(2 * math.log(max(L_max, 2))), L_max)
        top_scores, top_delays = torch.topk(scores, K_top, dim=-1)

        weights = F.softmax(top_scores / self.temperature, dim=-1)

        # Aggregate using original unpadded v and the target query length L_q
        mixed = aggregate_delays(v_original, top_delays, weights, L_q)

        mixed = mixed.permute(0, 3, 1, 2).reshape(B, L_q, D)
        return self.out(mixed)

class AutoformerEncoderBlock(nn.Module):
    """
    An encoder layer interleaving Auto-Correlation and Series Decomposition,
    with LayerNorm and a 4× expansion FFN.
    """
    def __init__(self, d_model, n_heads, moving_avg_kernel, dropout):
        super().__init__()
        self.mix = AutoCorrelationMixer(d_model, n_heads)
        self.decomp1 = SeriesDecomposition(moving_avg_kernel)
        self.decomp2 = SeriesDecomposition(moving_avg_kernel)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Mixer → decomposition (discard trend, keep remainder)
        x = x + self.drop(self.mix(self.norm1(x)))
        x, _ = self.decomp1(x)
        # FFN → decomposition
        x = x + self.drop(self.feed_forward(self.norm2(x)))
        x, _ = self.decomp2(x)
        return x


class AutoformerDecoderBlock(nn.Module):
    """
    A decoder layer with self-correlation, cross-correlation, FFN,
    and progressive trend accumulation through each sublayer.
    """
    def __init__(self, d_model, n_heads, c_out, moving_avg_kernel, dropout):
        super().__init__()
        # Self auto-correlation
        self.self_mix = AutoCorrelationMixer(d_model, n_heads)
        self.decomp1 = SeriesDecomposition(moving_avg_kernel)

        # Cross auto-correlation (queries from decoder, keys/values from encoder)
        self.cross_mix = AutoCorrelationMixer(d_model, n_heads)
        self.decomp2 = SeriesDecomposition(moving_avg_kernel)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.decomp3 = SeriesDecomposition(moving_avg_kernel)

        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Each sublayer's extracted trend is projected to c_out and accumulated
        self.trend_proj1 = nn.Linear(d_model, c_out, bias=False)
        self.trend_proj2 = nn.Linear(d_model, c_out, bias=False)
        self.trend_proj3 = nn.Linear(d_model, c_out, bias=False)

    def forward(self, x, cross, trend):
        """
        x:     [B, S, d]     – seasonal decoder stream
        cross: [B, L, d]     – encoder output
        trend: [B, S, c_out] – running trend accumulator
        """
        # Self auto-correlation + decomposition
        x = x + self.drop(self.self_mix(self.norm1(x)))
        x, trend1 = self.decomp1(x)

        # Cross auto-correlation + decomposition
        x = x + self.drop(self.cross_mix(self.norm2(x), cross))
        x, trend2 = self.decomp2(x)

        # FFN + decomposition
        x = x + self.drop(self.feed_forward(self.norm3(x)))
        x, trend3 = self.decomp3(x)

        # Accumulate trend from all three sublayers
        trend = (trend
                 + self.trend_proj1(trend1)
                 + self.trend_proj2(trend2)
                 + self.trend_proj3(trend3))
        return x, trend


class Autoformer(nn.Module):
    """
    Full Autoformer with:
      • Encoder–decoder architecture with cross auto-correlation
      • Progressive trend accumulation across decoder layers
      • RevIN (per-window normalization) for non-stationary series
      • Future covariate injection into the decoder
      • LayerNorm, 4× FFN expansion, and wider embedding convolution
    """
    def __init__(self, context=96, horizon=168, label_len=48,
                 in_channels=1, d=64, heads=4,
                 enc_layers=2, dec_layers=1,
                 dropout=0.1, kernel=25, c_out=1,
                 num_future_cov=0):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.label_len = label_len
        self.in_channels = in_channels
        self.c_out = c_out
        self.num_future_cov = num_future_cov

        # RevIN for the target channel
        self.revin = RevIN(num_features=1)

        # Initial decomposition
        self.decomp = SeriesDecomposition(kernel)

        # Encoder value embedding (wider kernel for more local context)
        self.enc_embed = nn.Conv1d(in_channels, d, kernel_size=7, padding=3,
                                   padding_mode="circular", bias=False)
        self.enc_pos = nn.Parameter(torch.randn(1, context, d) * 0.02)
        self.enc_drop = nn.Dropout(dropout)

        # Decoder value embedding
        self.dec_embed = nn.Conv1d(in_channels, d, kernel_size=7, padding=3,
                                   padding_mode="circular", bias=False)
        self.dec_pos = nn.Parameter(torch.randn(1, label_len + horizon, d) * 0.02)
        self.dec_drop = nn.Dropout(dropout)

        # Future covariate projection (added to decoder embedding at horizon positions)
        if num_future_cov > 0:
            self.cov_proj = nn.Linear(num_future_cov, d)

        # Encoder
        self.encoder = nn.ModuleList([
            AutoformerEncoderBlock(d, heads, kernel, dropout)
            for _ in range(enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d)

        # Decoder
        self.decoder = nn.ModuleList([
            AutoformerDecoderBlock(d, heads, c_out, kernel, dropout)
            for _ in range(dec_layers)
        ])

        # Final seasonal-to-output projection
        self.seasonal_proj = nn.Linear(d, c_out)

    def forward(self, x_enc, x_dec_cov=None):
        """
        x_enc:     [B, L, C]               – context window (target + covariates)
        x_dec_cov: [B, H, num_future_cov]  – future covariates over horizon (optional)
        Returns:   [B, H]                  – point forecasts for the horizon
        """
        B, L, C = x_enc.shape
        device = x_enc.device

        # ---- RevIN: per-window normalize the target channel ----
        target = x_enc[:, :, 0:1]                            # [B, L, 1]
        target = self.revin(target, 'norm')
        x_enc = torch.cat([target, x_enc[:, :, 1:]], dim=-1) if C > 1 else target

        # ---- Initial decomposition ----
        seasonal_enc, trend_enc = self.decomp(x_enc)

        # ---- Encoder ----
        enc_emb = self.enc_embed(seasonal_enc.transpose(1, 2)).transpose(1, 2)
        enc_emb = self.enc_drop(enc_emb + self.enc_pos)

        enc_out = enc_emb
        for block in self.encoder:
            enc_out = block(enc_out)
        enc_out = self.enc_norm(enc_out)                     # [B, L, d]

        # ---- Decoder initialization ----
        S = self.label_len + self.horizon

        # Seasonal: last label_len of decomposed seasonal + zeros for horizon
        seasonal_label = seasonal_enc[:, -self.label_len:, :]           # [B, label_len, C]
        seasonal_zeros = torch.zeros(B, self.horizon, C, device=device)
        seasonal_init = torch.cat([seasonal_label, seasonal_zeros], dim=1)  # [B, S, C]

        # Trend: last label_len of trend + context mean extended for horizon
        trend_label = trend_enc[:, -self.label_len:, 0:1]               # [B, label_len, 1]
        ctx_mean = x_enc[:, :, 0:1].mean(dim=1, keepdim=True)          # [B, 1, 1]
        trend_pred = ctx_mean.expand(B, self.horizon, 1)                # [B, H, 1]
        trend_init = torch.cat([trend_label, trend_pred], dim=1)        # [B, S, 1]

        # ---- Decoder embedding ----
        dec_emb = self.dec_embed(seasonal_init.transpose(1, 2)).transpose(1, 2)
        dec_emb = self.dec_drop(dec_emb + self.dec_pos)

        # Inject future covariates into the horizon portion of the decoder
        if x_dec_cov is not None and self.num_future_cov > 0:
            cov_pad = torch.zeros(B, self.label_len, self.num_future_cov, device=device)
            cov_full = torch.cat([cov_pad, x_dec_cov], dim=1)           # [B, S, num_cov]
            dec_emb = dec_emb + self.cov_proj(cov_full)

        # ---- Decoder forward (progressive trend accumulation) ----
        seasonal_out = dec_emb
        trend_out = trend_init                                          # [B, S, 1]
        for block in self.decoder:
            seasonal_out, trend_out = block(seasonal_out, enc_out, trend_out)

        # ---- Final output: horizon portion only ----
        seasonal_final = self.seasonal_proj(seasonal_out[:, -self.horizon:, :])  # [B, H, 1]
        output = (trend_out[:, -self.horizon:, :] + seasonal_final).squeeze(-1) # [B, H]

        # ---- RevIN denormalize back to input scale ----
        output = self.revin(output.unsqueeze(-1), 'denorm').squeeze(-1)

        return output

    @property
    def parameter_count(self):
        """
        Returns the total number of trainable parameters as required for the leaderboard.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
