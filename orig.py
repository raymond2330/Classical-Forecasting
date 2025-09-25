import time, random, math
from typing import List
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1337)

# ----------------------------
# Dataset
# ----------------------------
class LoadWindowDataset(Dataset):
    def __init__(self, data_array, window, horizon):
        self.data = data_array
        self.window, self.horizon = window, horizon
        self.n = data_array.shape[0] - window - horizon + 1
        if self.n <= 0:
            raise ValueError("Window/horizon too large for data length.")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window]                                        # [W, F]
        y_seq = self.data[idx + self.window: idx + self.window + self.horizon, -1:] # [H, 1]
        return torch.from_numpy(x).float(), torch.from_numpy(y_seq).float()

# ----------------------------
# Model
# ----------------------------
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

# ----------------------------
# Early Stopping (CPU snapshot)
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=25, delta=1e-4, verbose=True):
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
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: stop after {self.patience} epochs.")
        return improved

# ----------------------------
# Helpers
# ----------------------------
def infer_5min_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    freq = pd.infer_freq(index)
    if freq is None:
        diffs = pd.Series(index[1:] - index[:-1])
        step = diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(minutes=5)
    else:
        step = pd.Timedelta(freq)
    if step != pd.Timedelta(minutes=5):
        print(f"[WARN] Detected step={step}. Script assumes 5-minute cadence.")
    return step

def make_calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    hour = idx.hour + idx.minute/60.0
    hod = hour / 24.0
    dow = idx.dayofweek.astype(float) / 7.0
    doy = (idx.dayofyear.astype(float) - 1.0) / 366.0
    cal = pd.DataFrame(index=idx)
    cal["hod_sin"] = np.sin(2*np.pi*hod); cal["hod_cos"] = np.cos(2*np.pi*hod)
    cal["dow_sin"] = np.sin(2*np.pi*dow); cal["dow_cos"] = np.cos(2*np.pi*dow)
    cal["doy_sin"] = np.sin(2*np.pi*doy); cal["doy_cos"] = np.cos(2*np.pi*doy)
    return cal

def cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / float(warmup_epochs)
        denom = max(float(total_epochs - warmup_epochs), 1.0)
        progress = min(max((epoch - warmup_epochs) / denom, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------------------
# Main
# ----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load
    df = pd.read_csv('Demand_smoothed_kalman.csv', parse_dates=['DateTime'], index_col='DateTime').fillna(0)
    df.index = pd.to_datetime(df.index, format='mixed', errors='raise')

    # Impute outages causally (mark & fill)
    if 'Outage' in df.columns:
        df.loc[df['Outage'] == 1, 'TOTAL'] = pd.NA
    df['TOTAL'] = df['TOTAL'].interpolate(method='time')
    df['TOTAL'] = df['TOTAL'].bfill().ffill()

    # Cadence sanity
    _ = infer_5min_freq(df.index)

    # Window sizes
    W = 7 * 24 * 12   # one week lookback (7 days) - 5 minute resolution
    H = 1 * 24 * 12   # 24h horizon - 5 minute resolution

    # ----------------------------
    # Feature engineering (LEAK-FREE)
    # ----------------------------
    steps_per_hour = 12
    steps_1h  = 1  * steps_per_hour       # 12
    steps_6h  = 6  * steps_per_hour       # 72
    steps_24h = 24 * steps_per_hour       # 288

    # Seasonality lags requested (5-min steps): ~1.7d, ~6 months, ~1 year
    season_lags = [52520]
    #season_lags = [500, 52520, 105040]
    for L in season_lags:
        df[f'TOTAL_lag{L}'] = df['TOTAL'].shift(L)

    # Robust sudden drop detector (ABS + REL rules), made PAST-ONLY with shift(1)
    win_7d = 7 * 24 * 12
    k_abs  = 2.5
    k_pct  = 0.25

    diff = df['TOTAL'].diff()
    pct  = df['TOTAL'].pct_change()
    roll_std = diff.rolling(win_7d, min_periods=win_7d//2).std()

    abs_rule = diff <= (-k_abs * roll_std.fillna(diff.std()))
    rel_rule = pct  <= (-k_pct)

    sudden_drop_raw = (abs_rule | rel_rule).astype(int)
    df['sudden_drop'] = sudden_drop_raw.shift(1).fillna(0).astype(int)

    # Drop magnitude (past-only)
    drop_mag_raw = np.clip(-diff, 0.0, None)
    df['drop_mag'] = drop_mag_raw.shift(1).fillna(0.0)

    # Persistence lags (all past-only)
    drop_lags = [1, 2, 3, 6, 12]
    for L in drop_lags:
        df[f'sudden_drop_lag{L}'] = df['sudden_drop'].shift(L).fillna(0).astype(int)
        df[f'drop_mag_lag{L}']    = df['drop_mag'].shift(L).fillna(0.0)

    # Rolling stats on shifted series (past-only)
    total_shifted = df['TOTAL'].shift(1)
    df['tot_mean_1h']  = total_shifted.rolling(steps_1h,  min_periods=steps_1h//2).mean()
    df['tot_std_1h']   = total_shifted.rolling(steps_1h,  min_periods=steps_1h//2).std()
    df['tot_mean_6h']  = total_shifted.rolling(steps_6h,  min_periods=steps_6h//2).mean()
    df['tot_std_6h']   = total_shifted.rolling(steps_6h,  min_periods=steps_6h//2).std()
    df['tot_mean_24h'] = total_shifted.rolling(steps_24h, min_periods=steps_24h//2).mean()
    df['tot_std_24h']  = total_shifted.rolling(steps_24h, min_periods=steps_24h//2).std()

    # Safe momentum (past-only) with epsilon and clipping to avoid inf/huge values
    def safe_div(num, den, eps=1e-6):
        den = den.copy()
        # avoid exactly zero/near-zero denominators
        den = den.where(den.abs() > eps, eps)
        return num / den

    m1_num = df['TOTAL'].shift(1)
    m1_den = df['TOTAL'].shift(1 + steps_1h)
    m6_den = df['TOTAL'].shift(1 + steps_6h)

    df['mom_1h'] = safe_div(m1_num, m1_den) - 1.0
    df['mom_6h'] = safe_div(m1_num, m6_den) - 1.0
    df['mom_1h'] = df['mom_1h'].clip(-3, 3)
    df['mom_6h'] = df['mom_6h'].clip(-3, 3)

    # Calendar + exogenous features
    cal = make_calendar_features(df.index)
    exo_cols: List[str] = []
    if 'Temperature' in df.columns: exo_cols.append('Temperature')
    if 'Outage' in df.columns:      exo_cols.append('Outage')

    # Add policy and calendar binaries as exogenous
    policy_cols = ['Holiday', 'ECQ', 'GCQ', 'ALERT LEVEL 1', 'ALERT LEVEL 2', 'ALERT LEVEL 3']
    weekday_cols = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    for col in policy_cols + weekday_cols:
        if col in df.columns:
            exo_cols.append(col)

    # Interactions (past-only w.r.t signals used)
    interaction_cols = []
    if 'Temperature' in df.columns:
        df['temp_x_drop1'] = df['Temperature'] * df['sudden_drop_lag1']
        df['temp_x_mom1h'] = df['Temperature'] * df['mom_1h']
        interaction_cols += ['temp_x_drop1', 'temp_x_mom1h']

    # Assemble engineered feature table
    lag_cols      = [f'TOTAL_lag{L}' for L in season_lags]
    drop_lag_cols = [f'sudden_drop_lag{L}' for L in drop_lags] + [f'drop_mag_lag{L}' for L in drop_lags]
    roll_cols     = ['tot_mean_1h','tot_std_1h','tot_mean_6h','tot_std_6h','tot_mean_24h','tot_std_24h']
    mom_cols      = ['mom_1h','mom_6h']

    feat_blocks = [cal]
    if exo_cols:         feat_blocks.append(df[exo_cols])
    feat_blocks.append(df[lag_cols])
    feat_blocks.append(df[['sudden_drop','drop_mag'] + drop_lag_cols])
    feat_blocks.append(df[roll_cols + mom_cols])
    if interaction_cols:
        feat_blocks.append(df[interaction_cols])

    X_exo = pd.concat(feat_blocks, axis=1)

    # ----------------------------
    # Finite-only sanitization BEFORE scaling
    # ----------------------------
    X_exo.replace([np.inf, -np.inf], np.nan, inplace=True)
    finite_rows = np.isfinite(X_exo.to_numpy()).all(axis=1)
    finite_target = np.isfinite(df['TOTAL'].to_numpy())
    valid_mask = finite_rows & finite_target

    df    = df.loc[valid_mask].copy()
    X_exo = X_exo.loc[valid_mask].copy()

    # Window end timestamps (each sample's last target time) AFTER filtering
    times = df.index[W + H - 1 :]

    # Splits
    train_end = pd.Timestamp('2023-12-25 23:59:59.971')
    val_end   = pd.Timestamp('2024-11-25 23:59:59.971')

    # ----------------------------
    # SCALING (leak-free)
    # ----------------------------
    train_mask_for_scaler = df.index <= train_end
    if not train_mask_for_scaler.any():
        raise ValueError("No rows on/before train_end available to fit the scaler.")

    feat_scaler = MinMaxScaler()

    # Clean TRAIN slice for fit
    X_train = X_exo.loc[train_mask_for_scaler].copy()
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    good_train_rows = np.isfinite(X_train.to_numpy()).all(axis=1)
    X_train = X_train.loc[good_train_rows]

    if X_train.empty:
        raise ValueError("No finite rows remain in training features after sanitization.")

    feat_scaler.fit(X_train.to_numpy())

    # Clean FULL for transform
    X_full = X_exo.copy()
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X_full.isna().any().any():
        X_full = X_full.bfill().ffill()

    keep_mask_full = np.isfinite(X_full.to_numpy()).all(axis=1)
    if not keep_mask_full.all():
        X_full = X_full.loc[keep_mask_full]
        df     = df.loc[X_full.index]
        times  = df.index[W + H - 1 :]

    X_feats_scaled = feat_scaler.transform(X_full.to_numpy())

    tgt_scaler = MinMaxScaler()
    tgt_scaler.fit(df.loc[train_mask_for_scaler, ['TOTAL']].to_numpy())
    y_scaled = tgt_scaler.transform(df[['TOTAL']].to_numpy())

    # Final data array: [features..., target_scaled] with target LAST
    data = np.hstack([X_feats_scaled, y_scaled])
    n_features = data.shape[1]  # = F_feat + 1

    # ----------------------------
    # Indexing with optional stride (speeds up training)
    # ----------------------------
    stride = 6
    all_idx = np.arange(len(times))
    train_idx = all_idx[times <= train_end][::stride]
    val_idx   = all_idx[(times > train_end) & (times <= val_end)][::stride]
    test_idx  = all_idx[times > val_end]  # keep full test (or add [::stride] to speed eval)

    print(f"Train windows: {len(train_idx)}, Validation windows: {len(val_idx)},  Test windows: {len(test_idx)}")
    if len(times) > 0:
        print(f"First window ts: {times.min()}, Last: {times.max()}")
    assert len(train_idx) > 0, "No training windows found—check train_end vs. data coverage"
    assert len(val_idx) > 0, "No validation windows found—check val_end vs. data coverage"
    assert len(test_idx) > 0, "No test windows found—does your data extend past val_end?"

    # Datasets / Loaders
    full_ds = LoadWindowDataset(data, W, H)
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=2, pin_memory=pin, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=2, pin_memory=pin, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                              num_workers=2, pin_memory=pin, persistent_workers=True)

    # Model / Optimizer / Loss / Scheduler
    model = LSTMAE_Forecaster(n_features=n_features, window=W, horizon=H,
        enc1=256, enc2=128, dec1=128, dec2=256, latent_dim=64, p_drop=0.2
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    es = EarlyStopping(patience=25, delta=1e-4, verbose=True)

    weights = torch.linspace(0.5, 1.5, steps=H, device=device).view(1, H, 1)
    def weighted_mse(pred, target):
        return ((weights * (pred - target) ** 2).mean())
    f_fn = weighted_mse

    total_epochs = 50
    warmup_epochs = 10
    sched = cosine_warmup_scheduler(opt, warmup_epochs, total_epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("No. of parameters:", n_params)

    train_losses, val_losses = [], []

    # ----------------------------
    # Training Loop
    # ----------------------------
    for ep in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        tl = 0.0
        for Xb, Yseqb in train_loader:
            Xb, Yseqb = Xb.to(device), Yseqb.to(device)
            yhat = model(Xb)
            loss = f_fn(yhat, Yseqb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * Xb.size(0)
        train_losses.append(tl / len(train_ds))

        # Validation
        model.eval()
        vl, preds_last_scaled, truths_last_scaled = 0.0, [], []
        preds_full_scaled, truths_full_scaled = [], []
        with torch.no_grad():
            for Xb, Yseqb in val_loader:
                Xb, Yseqb = Xb.to(device), Yseqb.to(device)
                yhat = model(Xb)
                vl += f_fn(yhat, Yseqb).item() * Xb.size(0)
                preds_last_scaled.append(yhat[:, -1, 0].cpu().numpy())
                truths_last_scaled.append(Yseqb[:, -1, 0].cpu().numpy())
                preds_full_scaled.append(yhat[:, :, 0].cpu().numpy())
                truths_full_scaled.append(Yseqb[:, :, 0].cpu().numpy())
        val_loss = vl / len(val_ds)
        val_losses.append(val_loss)

        # step LR ONCE PER EPOCH (important for LambdaLR)
        sched.step()

        # Inverse metrics
        preds = np.concatenate(preds_last_scaled)
        truths = np.concatenate(truths_last_scaled)
        inv_p = tgt_scaler.inverse_transform(preds.reshape(-1,1))[:,0]
        inv_t = tgt_scaler.inverse_transform(truths.reshape(-1,1))[:,0]
        rm  = root_mean_squared_error(inv_t, inv_p)
        mae = mean_absolute_error(inv_t, inv_p)
        r2  = r2_score(inv_t, inv_p)

        pf = np.concatenate(preds_full_scaled, axis=0)
        tf = np.concatenate(truths_full_scaled, axis=0)
        inv_pf = tgt_scaler.inverse_transform(pf.reshape(-1,1))[:,0]
        inv_tf = tgt_scaler.inverse_transform(tf.reshape(-1,1))[:,0]
        rm_all  = root_mean_squared_error(inv_tf, inv_pf)
        mae_all = mean_absolute_error(inv_tf, inv_pf)

        cur_lr = opt.param_groups[0]['lr']
        print(f"Ep{ep:02d} Train={train_losses[-1]:.4f} | Val={val_loss:.4f} | "
              f"Val(last) RMSE={rm:.1f} MAE={mae:.1f} R²={r2:.3f} | "
              f"Val(full) RMSE={rm_all:.1f} MAE={mae_all:.1f} | "
              f"LR={cur_lr:.6f} | Time={time.time()-t0:.1f}s")

        _ = es.step(val_loss, model=model)
        if es.should_stop:
            if es.best_state is not None:
                model.load_state_dict(es.best_state)
                if es.verbose:
                    print("EarlyStopping: best weights restored.")
            break

    # ----------------------------
    # Loss curves
    # ----------------------------
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Weighted MSE'); plt.legend(); plt.title('Loss Curves (tail-weighted)')
    plt.tight_layout(); plt.show()

    # ----------------------------
    # Testing
    # ----------------------------
    model.eval()
    preds_last_scaled, truths_last_scaled = [], []
    preds_full_scaled, truths_full_scaled = [], []
    test_vl = 0.0
    with torch.no_grad():
        for Xb, Yseqb in test_loader:
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
    inv_p = tgt_scaler.inverse_transform(preds.reshape(-1,1))[:,0]
    inv_t = tgt_scaler.inverse_transform(truths.reshape(-1,1))[:,0]

    mse = mean_squared_error(inv_t, inv_p)
    rm  = root_mean_squared_error(inv_t, inv_p)
    mae = mean_absolute_error(inv_t, inv_p)
    r2  = r2_score(inv_t, inv_p)
    n, p = len(inv_t), (n_features - 1)  # p ~ number of non-target features
    adj_r2 = 1 - (1-r2)*(n-1)/max(n-p-1, 1)

    PH = np.concatenate(preds_full_scaled, axis=0)
    TH = np.concatenate(truths_full_scaled, axis=0)
    inv_pf = tgt_scaler.inverse_transform(PH.reshape(-1,1))[:,0]
    inv_tf = tgt_scaler.inverse_transform(TH.reshape(-1,1))[:,0]
    rm_all  = root_mean_squared_error(inv_tf, inv_pf)
    mae_all = mean_absolute_error(inv_tf, inv_pf)

    # Per-lead RMSE (hourly average over 24h)
    lead_rmse = []
    for h in range(PH.shape[1]):
        inv_ph = tgt_scaler.inverse_transform(PH[:, h].reshape(-1,1))[:,0]
        inv_th = tgt_scaler.inverse_transform(TH[:, h].reshape(-1,1))[:,0]
        lead_rmse.append(root_mean_squared_error(inv_th, inv_ph))
    hourly_rmse = np.array(lead_rmse).reshape(-1, 12).mean(axis=1)

    print(f"Test Loss (tail-weighted MSE): {test_loss:.4f}")
    print(f"\nFinal Test Metrics (last step, actual units): "
          f"MSE={mse:.2f}, RMSE={rm:.2f}, MAE={mae:.2f}, R²={r2:.4f}, Adj R²={adj_r2:.4f}")
    print(f"Final Test Metrics (full 24h path): RMSE={rm_all:.2f}, MAE={mae_all:.2f}")
    print("Hourly-averaged RMSE by lead hour (0→23):")
    print(np.array2string(hourly_rmse, precision=2, separator=', '))

    # Inference latency (batch=1)
    _ = model(torch.randn(1, W, n_features).to(device))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    Nruns = 100
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    with torch.no_grad():
        for _ in range(Nruns):
            inp = torch.randn(1, W, n_features).to(device)
            start.record(); _ = model(inp); end.record()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += start.elapsed_time(end) / 1000.0
    avg_batch = total_time / Nruns
    print(f"Avg inference latency: {avg_batch*1000:.2f} ms/batch • {avg_batch*1000:.2f} ms/sample")

    # Plots
    plt.figure(figsize=(10,4))
    plt.plot(times[test_idx], inv_t, label='Actual')
    plt.plot(times[test_idx], inv_p, label='Predicted', alpha=0.7)
    plt.title("24h-Ahead Forecast – Last-Step (Test)")
    plt.xlabel("Timestamp"); plt.ylabel("TOTAL demand")
    plt.legend(); plt.tight_layout(); plt.show()

    num_show = min(3, len(test_idx))
    if num_show > 0:
        Xb, Yseqb = next(iter(DataLoader(test_ds, batch_size=num_show, shuffle=False)))
        Xb = Xb.to(device)
        with torch.no_grad():
            yhat = model(Xb).cpu().numpy()
        ytrue = Yseqb.numpy()
        for i in range(min(num_show, Xb.size(0))):
            pred_i = tgt_scaler.inverse_transform(yhat[i, :, 0].reshape(-1,1))[:,0]
            true_i = tgt_scaler.inverse_transform(ytrue[i, :, 0].reshape(-1,1))[:,0]
            end_ts = times[test_idx[i]]
            tgrid = pd.date_range(end=end_ts, periods=H, freq='5min')
            plt.figure(figsize=(10,3.5))
            plt.plot(tgrid, true_i, label='Actual')
            plt.plot(tgrid, pred_i, label='Predicted', alpha=0.8)
            plt.title(f"Full 24h Path – Sample {i} (ends {end_ts})")
            plt.xlabel("Timestamp"); plt.ylabel("TOTAL demand")
            plt.legend(); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
