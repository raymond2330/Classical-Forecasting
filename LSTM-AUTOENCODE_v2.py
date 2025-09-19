import time
from typing import Dict
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Dataset 
class LoadWindowDataset(Dataset):
    def __init__(self, data_array, window, horizon):
        self.data = data_array
        self.window, self.horizon = window, horizon
        self.n = data_array.shape[0] - window - horizon + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Input: past W steps
        x = self.data[idx: idx + self.window]  # [W, F]
        # Future H steps of TOTAL (last column)
        y_seq = self.data[idx + self.window: idx + self.window + self.horizon, -1:]  # [H, 1]
        y_last = y_seq[-1, 0]
        return torch.from_numpy(x).float(), torch.from_numpy(y_seq).float(), torch.tensor([y_last]).float()

# Model  — LSTM Autoencoder-style forecaster (drops recon in output to keep your loop unchanged)
class LSTMAE_Forecaster(nn.Module):
    def __init__(self, n_features, window, horizon,
                 enc1=256, enc2=128, dec1=128, dec2=256,
                 latent_dim=64, p_drop=0.2):
        super().__init__()
        self.n_features = n_features
        self.window = window
        self.horizon = horizon

        # Encoder
        self.enc1 = nn.LSTM(input_size=n_features, hidden_size=enc1, batch_first=True)
        self.enc2 = nn.LSTM(input_size=enc1,       hidden_size=enc2, batch_first=True)
        self.enc_ln = nn.LayerNorm(enc2)

        # Attention pooling over encoder sequence -> context
        self.attn = nn.Sequential(
            nn.Linear(enc2, enc2),
            nn.Tanh(),
            nn.Linear(enc2, 1)
        )

        # context -> latent
        self.to_latent = nn.Sequential(
            nn.Linear(enc2, latent_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(latent_dim, latent_dim)
        )

        # Decoder (reconstruction branch; not used in loss here)
        # kept to stay true to the AE architecture, but forward() returns only forecast yhat
        self.dec1 = nn.LSTM(input_size=latent_dim, hidden_size=dec1, batch_first=True)
        self.dec2 = nn.LSTM(input_size=dec1,       hidden_size=dec2, batch_first=True)
        self.dec_proj = nn.Sequential(
            nn.Linear(window * dec2, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, window * n_features)
        )

        # Forecast head (TOTAL for H steps)
        # condition on [latent || last_observed_features] to anchor short-term
        f_in = latent_dim + n_features
        self.forecast = nn.Sequential(
            nn.Linear(f_in, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, horizon)  # -> [B, H]
        )

    def forward(self, x):
        """
        x: [B, W, F]
        return: yhat [B, H, 1]  (to fit your existing training loop)
        """
        B, W, F = x.shape

        # Encode
        h1, _ = self.enc1(x)               # [B,W,enc1]
        h2, _ = self.enc2(h1)              # [B,W,enc2]
        h2 = self.enc_ln(h2)

        # Attention pooling across time
        a = self.attn(h2)                  # [B,W,1]
        a = torch.softmax(a, dim=1)
        ctx = (a * h2).sum(dim=1)          # [B,enc2]

        # Latent
        z = self.to_latent(ctx)            # [B,latent]

        # Forecast
        x_last = x[:, -1, :]               # [B,F]
        f_in = torch.cat([z, x_last], dim=1)   # [B,latent+F]
        yhat = self.forecast(f_in).unsqueeze(-1)  # [B,H,1]
        return yhat

# Early Stopping 
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False):
        self.patience, self.delta, self.verbose = patience, delta, verbose
        self.best_loss = np.inf
        self.wait = 0
        self.should_stop = False
        self.best_state = None

    def step(self, val_loss, model=None):
        improved = val_loss < self.best_loss - self.delta
        if improved:
            self.best_loss = val_loss
            self.wait = 0
            if model is not None:
                self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: stop after {self.patience} epochs.")
        return improved

# Helpers for cadence & per-year target scaling
def infer_5min_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Return the expected 5-minute step; warn if index not uniform 5T."""
    freq = pd.infer_freq(index)
    if freq is None:
        diffs = pd.Series(index[1:] - index[:-1])
        step = diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(minutes=5)
    else:
        step = pd.Timedelta(freq)
    if step != pd.Timedelta(minutes=5):
        print(f"[WARN] Detected step={step}. Script assumes 5-minute cadence.")
    return step

def fit_target_scalers_per_year(df: pd.DataFrame, target_col: str) -> Dict[int, MinMaxScaler]:
    """Fit a separate MinMaxScaler for target on each calendar year (using all rows in that year)."""
    scalers = {}
    years = df.index.year
    for yr in np.unique(years):
        mask = (years == yr)
        sc = MinMaxScaler()
        sc.fit(df.loc[mask, [target_col]])
        scalers[int(yr)] = sc
    return scalers

def transform_target_per_year(df: pd.DataFrame, target_col: str, scalers_by_year: Dict[int, MinMaxScaler]) -> np.ndarray:
    """Transform target to scaled values using its own year's scaler. Returns (N,1)."""
    years = df.index.year
    y_scaled = np.empty((len(df), 1), dtype=float)
    for yr, sc in scalers_by_year.items():
        mask = (years == yr)
        if mask.any():
            y_scaled[mask, :] = sc.transform(df.loc[mask, [target_col]])
    return y_scaled

def inverse_target_laststep(y_scaled_1d: np.ndarray, years_for_rows: np.ndarray, scalers_by_year: Dict[int, MinMaxScaler]) -> np.ndarray:
    """Inverse-transform last-step scaled target values using the scaler for that last timestamp's year."""
    assert y_scaled_1d.ndim == 1
    assert y_scaled_1d.shape[0] == years_for_rows.shape[0]
    y_inv = np.empty_like(y_scaled_1d, dtype=float)
    for yr in np.unique(years_for_rows):
        sc = scalers_by_year[int(yr)]
        mask = (years_for_rows == yr)
        y_inv[mask] = sc.inverse_transform(y_scaled_1d[mask].reshape(-1,1))[:,0]
    return y_inv

# Main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load
    df = pd.read_csv('Demand_with_Temperature.csv', parse_dates=['DateTime'], index_col='DateTime').fillna(0)
    df.index = pd.to_datetime(df.index, format='mixed', errors='raise')

    # Cadence sanity (expect 5-minute)
    _ = infer_5min_freq(df.index)

    # 48h lookback & 48h ahead at 5min => 576
    W = H = 48 * 12

    # Window end timestamps (each sample's last target time)
    times = df.index[W + H - 1 :]

    # Features: all except TOTAL, then TOTAL as last column
    feats = [c for c in df.columns if c == 'TOTAL'] # univariate for now 
    # feats = [c for c in df.columns if c != 'TOTAL'] + ['TOTAL']
    X_all = df[feats].astype(float)
    target_col = 'TOTAL'

    # Splits
    train_end = pd.Timestamp('2023-12-25 23:59:59.971')
    val_end   = pd.Timestamp('2024-11-25 23:59:59.971')

    # SCALING (features global; target per-year)
    train_mask_for_scaler = df.index <= train_end
    if not train_mask_for_scaler.any():
        raise ValueError("No rows on/before train_end available to fit the scaler.")

    feat_cols = feats[:-1]
    tgt_col   = feats[-1]  # 'TOTAL'

    # 1) Global feature scaler (fit on train rows only); transform all rows
    if len(feat_cols) > 0:
        feat_scaler = MinMaxScaler()
        feat_scaler.fit(df.loc[train_mask_for_scaler, feat_cols].values)
        X_feats_scaled = feat_scaler.transform(df[feat_cols].values)
    else:
        # Univariate case: no exogenous features
        X_feats_scaled = np.empty((len(df), 0), dtype=float)

    # 2) Per-year target scalers; transform all rows
    tgt_scalers_by_year = fit_target_scalers_per_year(df, tgt_col)
    y_scaled = transform_target_per_year(df, tgt_col, tgt_scalers_by_year)  # [N,1]

    # 3) Final data array for dataset: [features..., target_scaled] with target LAST
    data = np.hstack([X_feats_scaled, y_scaled])
    n_features = data.shape[1]

    # Indices for samples (by window last time)
    train_idx = np.where(times <= train_end)[0]
    val_idx   = np.where((times > train_end) & (times <= val_end))[0]
    test_idx  = np.where(times > val_end)[0]

    print(f"Train windows: {len(train_idx)}, Validation windows: {len(val_idx)},  Test windows: {len(test_idx)}")
    if len(times) > 0:
        print(f"First window ts: {times.min()}, Last: {times.max()}")
    assert len(train_idx) > 0, "No training windows found—check train_end vs. data coverage"
    assert len(val_idx) > 0, "No validation windows found—check val_end vs. data coverage"
    assert len(test_idx) > 0, "No test windows found—does your data extend past val_end?"

    # Datasets / Loaders
    full_ds = LoadWindowDataset(data, W, H)
    train_ds_pre  = Subset(full_ds, train_idx)
    train_ds_post = Subset(full_ds, val_idx)
    test_ds       = Subset(full_ds, test_idx)

    train_loader_pre  = DataLoader(train_ds_pre,  batch_size=32, shuffle=True,  num_workers=2)
    train_loader_post = DataLoader(train_ds_post, batch_size=32, shuffle=False, num_workers=2)  # keep order
    test_loader       = DataLoader(test_ds,       batch_size=32, shuffle=False, num_workers=2)

    # Model + optimizer + tail-weighted loss
    model = LSTMAE_Forecaster(n_features=len(feats), window=W, horizon=H,
        enc1=256, enc2=128, dec1=128, dec2=256, latent_dim=64, p_drop=0.2
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    es = EarlyStopping(patience=8, delta=1e-4, verbose=True)

    weights = torch.linspace(0.5, 1.5, steps=H, device=device).view(1, H, 1)
    def weighted_mse(pred, target):
        return ((weights * (pred - target) ** 2).mean())
    f_fn = weighted_mse

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, cooldown=1, min_lr=1e-5
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("No. of parameters:", n_params)

    train_losses, val_losses = [], []

    # Training Loop
    for ep in range(1, 31):
        t0 = time.time()
        model.train()
        tl = 0.0
        for Xb, Yseqb, _ in train_loader_pre:
            Xb, Yseqb = Xb.to(device), Yseqb.to(device)   # Xb:[B,W,F], Y:[B,H,1]
            yhat = model(Xb)                              # [B,H,1]
            loss = f_fn(yhat, Yseqb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * Xb.size(0)
        train_losses.append(tl / len(train_ds_pre))

        # Validation Loop
        model.eval()
        vl, preds_last_scaled, truths_last_scaled = 0.0, [], []
        preds_full_scaled, truths_full_scaled = [], []
        with torch.no_grad():
            for Xb, Yseqb, _ in train_loader_post:
                Xb, Yseqb = Xb.to(device), Yseqb.to(device)
                yhat = model(Xb)
                vl += f_fn(yhat, Yseqb).item() * Xb.size(0)
                preds_last_scaled.append(yhat[:, -1, 0].cpu().numpy())
                truths_last_scaled.append(Yseqb[:, -1, 0].cpu().numpy())
                preds_full_scaled.append(yhat[:, :, 0].cpu().numpy())
                truths_full_scaled.append(Yseqb[:, :, 0].cpu().numpy())
        val_loss = vl / len(train_ds_post)
        val_losses.append(val_loss)
        sched.step(val_loss)

        # Inverse to ACTUAL units (per-year target inverse)
        preds = np.concatenate(preds_last_scaled)   # scaled
        truths = np.concatenate(truths_last_scaled) # scaled
        val_years_last = times[val_idx].year.values
        inv_p = inverse_target_laststep(preds,  val_years_last, tgt_scalers_by_year)
        inv_t = inverse_target_laststep(truths, val_years_last, tgt_scalers_by_year)

        rm  = root_mean_squared_error(inv_t, inv_p)
        mae = mean_absolute_error(inv_t, inv_p)
        r2  = r2_score(inv_t, inv_p)

        # Full-horizon metrics (flatten) — approximate inverse with last-step year per sample
        pf = np.concatenate(preds_full_scaled, axis=0)  # [N,H] scaled
        tf = np.concatenate(truths_full_scaled, axis=0) # [N,H] scaled
        # Repeat each sample's last-year across H leads for inverse (48h rarely crosses year)
        yrs = np.repeat(val_years_last, H)
        inv_pf = inverse_target_laststep(pf.reshape(-1), yrs, tgt_scalers_by_year)
        inv_tf = inverse_target_laststep(tf.reshape(-1), yrs, tgt_scalers_by_year)
        rm_all  = root_mean_squared_error(inv_tf, inv_pf)
        mae_all = mean_absolute_error(inv_tf, inv_pf)

        print(f"Ep{ep:02d} Train={train_losses[-1]:.4f} | Val={val_loss:.4f} | "
              f"Val(last) RMSE={rm:.1f} MAE={mae:.1f} R²={r2:.3f} | "
              f"Val(full) RMSE={rm_all:.1f} MAE={mae_all:.1f} | "
              f"Time={time.time()-t0:.1f}s")

        _ = es.step(val_loss, model=model)
        if es.should_stop:
            if es.best_state is not None:
                model.load_state_dict(es.best_state)
                if es.verbose:
                    print("EarlyStopping: best weights restored.")
            break

    # Loss curves
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Weighted MSE'); plt.legend(); plt.title('Loss Curves (tail-weighted)')
    plt.tight_layout(); plt.show()

    # Testing Loop
    model.eval()
    preds_last_scaled, truths_last_scaled = [], []
    preds_full_scaled, truths_full_scaled = [], []
    test_vl = 0.0
    with torch.no_grad():
        for Xb, Yseqb, _ in DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2):
            Xb, Yseqb = Xb.to(device), Yseqb.to(device)
            yhat = model(Xb)
            test_vl += f_fn(yhat, Yseqb).item() * Xb.size(0)
            preds_last_scaled.append(yhat[:, -1, 0].cpu().numpy())
            truths_last_scaled.append(Yseqb[:, -1, 0].cpu().numpy())
            preds_full_scaled.append(yhat[:, :, 0].cpu().numpy())
            truths_full_scaled.append(Yseqb[:, :, 0].cpu().numpy())
    test_loss = test_vl / len(test_ds)

    preds = np.concatenate(preds_last_scaled)
    truths = np.concatenate(truths_last_scaled)
    test_years_last = times[test_idx].year.values
    inv_p = inverse_target_laststep(preds,  test_years_last, tgt_scalers_by_year)
    inv_t = inverse_target_laststep(truths, test_years_last, tgt_scalers_by_year)

    mse = mean_squared_error(inv_t, inv_p)
    rm  = root_mean_squared_error(inv_t, inv_p)
    mae = mean_absolute_error(inv_t, inv_p)
    r2  = r2_score(inv_t, inv_p)
    n, p = len(inv_t), len(feats)-1
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

    # Full-horizon test (approx inverse with last-step year per sample)
    PH = np.concatenate(preds_full_scaled, axis=0)  # [N,H]
    TH = np.concatenate(truths_full_scaled, axis=0) # [N,H]
    yrs = np.repeat(test_years_last, H)
    inv_pf = inverse_target_laststep(PH.reshape(-1), yrs, tgt_scalers_by_year)
    inv_tf = inverse_target_laststep(TH.reshape(-1), yrs, tgt_scalers_by_year)
    rm_all  = root_mean_squared_error(inv_tf, inv_pf)
    mae_all = mean_absolute_error(inv_tf, inv_pf)

    # Per-lead RMSE (hourly average), using same last-year approximation
    lead_rmse = []
    N = PH.shape[0]
    for h in range(PH.shape[1]):
        inv_ph = inverse_target_laststep(PH[:, h], test_years_last, tgt_scalers_by_year)
        inv_th = inverse_target_laststep(TH[:, h], test_years_last, tgt_scalers_by_year)
        lead_rmse.append(root_mean_squared_error(inv_th, inv_ph))
    hourly_rmse = np.array(lead_rmse).reshape(-1, 12).mean(axis=1)  # length 48

    print(f"Test Loss (tail-weighted MSE): {test_loss:.4f}")
    print(f"\nFinal Test Metrics (last step, actual units): "
          f"MSE={mse:.2f}, RMSE={rm:.2f}, MAE={mae:.2f}, R²={r2:.4f}, Adj R²={adj_r2:.4f}")
    print(f"Final Test Metrics (full 48h path): RMSE={rm_all:.2f}, MAE={mae_all:.2f}")
    print("Hourly-averaged RMSE by lead hour (0→47):")
    print(np.array2string(hourly_rmse, precision=2, separator=', '))

    # Inference latency (batch=1)
    _ = model(torch.randn(1, W, len(feats)).to(device))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    Nruns = 100
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    with torch.no_grad():
        for _ in range(Nruns):
            inp = torch.randn(1, W, len(feats)).to(device)
            start.record(); _ = model(inp); end.record()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += start.elapsed_time(end) / 1000.0
    avg_batch = total_time / Nruns
    print(f"Avg inference latency: {avg_batch*1000:.2f} ms/batch • {avg_batch*1000:.2f} ms/sample")

    # Plot test last-step series
    plt.figure(figsize=(10,4))
    plt.plot(times[test_idx], inv_t, label='Actual')
    plt.plot(times[test_idx], inv_p, label='Predicted', alpha=0.7)
    plt.title("48h-Ahead Forecast – Last-Step (Test)")
    plt.xlabel("Timestamp"); plt.ylabel("TOTAL demand")
    plt.legend(); plt.tight_layout(); plt.show()

    # Plot a few full 48h trajectories (prediction vs truth) — use last-year inverse per sample
    num_show = min(3, len(test_idx))
    if num_show > 0:
        Xb, Yseqb, _ = next(iter(DataLoader(test_ds, batch_size=num_show, shuffle=False)))
        Xb = Xb.to(device)
        with torch.no_grad():
            yhat = model(Xb).cpu().numpy()  # [B,H,1]
        ytrue = Yseqb.numpy()              # [B,H,1]
        for i in range(min(num_show, Xb.size(0))):
            last_year = int(times[test_idx[i]].year)
            sc = tgt_scalers_by_year[last_year]
            pred_i = sc.inverse_transform(yhat[i, :, 0].reshape(-1,1))[:,0]
            true_i = sc.inverse_transform(ytrue[i, :, 0].reshape(-1,1))[:,0]
            end_ts = times[test_idx[i]]
            tgrid = pd.date_range(end=end_ts, periods=H, freq='5min')
            plt.figure(figsize=(10,3.5))
            plt.plot(tgrid, true_i, label='Actual')
            plt.plot(tgrid, pred_i, label='Predicted', alpha=0.8)
            plt.title(f"Full 48h Path – Sample {i} (ends {end_ts})")
            plt.xlabel("Timestamp"); plt.ylabel("TOTAL demand")
            plt.legend(); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
