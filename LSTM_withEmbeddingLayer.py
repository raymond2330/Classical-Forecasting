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
# Dataset (now supports continuous + categorical inputs)
# ----------------------------
class LoadWindowDataset(Dataset):
    def __init__(self, cont_array, cat_array, y_array, window, horizon):
        """
        cont_array: [T, F_cont]  (continuous features only)
        cat_array:  [T, C]       (integer categorical features)
        y_array:    [T, 1]       (target scaled)
        """
        assert cont_array.shape[0] == cat_array.shape[0] == y_array.shape[0]
        self.cont = cont_array
        self.cat = cat_array
        self.y = y_array
        self.window, self.horizon = window, horizon
        self.n = cont_array.shape[0] - window - horizon + 1
        if self.n <= 0:
            raise ValueError("Window/horizon too large for data length.")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_cont = self.cont[idx: idx + self.window]                            # [W, F_cont]
        x_cat  = self.cat[idx: idx + self.window]                             # [W, C]
        y_seq  = self.y[idx + self.window: idx + self.window + self.horizon]  # [H, 1]
        return torch.from_numpy(x_cont).float(), torch.from_numpy(x_cat).long(), torch.from_numpy(y_seq).float()

# ----------------------------
# Model (embeddings for categorical features, concatenated to continuous inputs)
# ----------------------------
class LSTM_Forecaster(nn.Module):
    def __init__(self, n_cont_features, cat_vocab_sizes: List[int], window, horizon,
                 enc1=256, enc2=128, latent_dim=64, p_drop=0.2):
        """
        n_cont_features: number of continuous features per timestep
        cat_vocab_sizes: list of vocab sizes for each categorical input (e.g., [7,2,2])
        """
        super().__init__()
        self.n_cont = n_cont_features
        self.cat_vocab_sizes = cat_vocab_sizes
        self.window = window
        self.horizon = horizon

        # Embedding layers for each categorical column
        self.embed_layers = nn.ModuleList()
        self.embed_dims = []
        for vs in cat_vocab_sizes:
            # simple heuristic for embedding dim
            emb_dim = max(1, min(50, (vs + 1) // 2))
            self.embed_layers.append(nn.Embedding(vs, emb_dim))
            self.embed_dims.append(emb_dim)
        total_emb_dim = sum(self.embed_dims)

        # After concatenation per-timestep features = cont + embeddings
        input_size = self.n_cont + total_emb_dim

        # Encoder
        self.enc1 = nn.LSTM(input_size=input_size, hidden_size=enc1, batch_first=True)
        self.enc2 = nn.LSTM(input_size=enc1, hidden_size=enc2, batch_first=True)
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

        # Forecast head (TOTAL for H steps)
        # condition on [latent || last_observed_continuous + last_embeds] to anchor short-term
        f_in = latent_dim + self.n_cont + total_emb_dim
        self.forecast = nn.Sequential(
            nn.Linear(f_in, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, horizon)  # -> [B, H]
        )

    def forward(self, x_cont, x_cat):
        """
        x_cont: [B, W, F_cont]
        x_cat:  [B, W, C]   (long/int indices)
        return: yhat [B, H, 1]
        """
        B, W, F = x_cont.shape
        # build embeddings per timestep
        emb_parts = []
        for i, emb_layer in enumerate(self.embed_layers):
            emb_i = emb_layer(x_cat[:, :, i])   # [B, W, emb_dim_i]
            emb_parts.append(emb_i)
        if len(emb_parts) > 0:
            emb = torch.cat(emb_parts, dim=2)   # [B, W, total_emb_dim]
            enc_in = torch.cat([x_cont, emb], dim=2)  # [B, W, input_size]
        else:
            enc_in = x_cont

        # Encode
        h1, _ = self.enc1(enc_in)               # [B,W,enc1]
        h2, _ = self.enc2(h1)                   # [B,W,enc2]
        h2 = self.enc_ln(h2)

        # Attention pooling across time
        a = self.attn(h2)                       # [B,W,1]
        a = torch.softmax(a, dim=1)
        ctx = (a * h2).sum(dim=1)               # [B,enc2]

        # Latent
        z = self.to_latent(ctx)                 # [B,latent]

        # Forecast head uses latent + last timestep (cont + embeds) for anchoring
        x_last_cont = x_cont[:, -1, :]          # [B, F_cont]
        if len(emb_parts) > 0:
            x_last_emb = emb[:, -1, :]          # [B, total_emb_dim]
            f_in = torch.cat([z, x_last_cont, x_last_emb], dim=1)  # [B, latent + F_cont + emb]
        else:
            f_in = torch.cat([z, x_last_cont], dim=1)

        yhat = self.forecast(f_in).unsqueeze(-1)  # [B, H, 1]
        return yhat

# ----------------------------
# Early Stopping (CPU snapshot)
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4, verbose=True):
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
# Helpers (unchanged)
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

def mean_absolute_percentage_error(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.clip(np.abs(y_true), eps, None)))) * 100

def smape(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )

# ----------------------------
# Main (with embedding preprocessing)
# ----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load
    df = pd.read_csv('Dataset/Demand_smoothed_kalman.csv', parse_dates=['DateTime'], index_col='DateTime').fillna(0)
    df.index = pd.to_datetime(df.index, format='mixed', errors='raise')

    # Cadence sanity
    _ = infer_5min_freq(df.index)

    # Window sizes
    W = 7 * 24 * 12   # one week lookback (7 days) - 5 minute resolution
    H = 1 * 24 * 12   # 24h horizon - 5 minute resolution

    # --- BUILD FEATURES (same as your code) ---
    # ... (copy-paste your entire feature engineering block here, exactly as you had it)
    # For brevity in this snippet, I'm using the same exact feature construction you provided above.
    # (In your file, keep the full feature engineering code you already have up to "X_exo = pd.concat(...)")
    # -------------------------------------------------------------------------
    # (Start of your feature engineering block)
    steps_per_hour = 12
    steps_1h  = 1  * steps_per_hour       # 12
    steps_6h  = 6  * steps_per_hour       # 72
    steps_24h = 24 * steps_per_hour       # 288

    season_lags = [500]
    for L in season_lags:
        df[f'TOTAL_lag{L}'] = df['TOTAL'].shift(L)

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

    drop_mag_raw = np.clip(-diff, 0.0, None)
    df['drop_mag'] = drop_mag_raw.shift(1).fillna(0.0)

    drop_lags = [1, 2, 3, 6, 12]
    for L in drop_lags:
        df[f'sudden_drop_lag{L}'] = df['sudden_drop'].shift(L).fillna(0).astype(int)
        df[f'drop_mag_lag{L}']    = df['drop_mag'].shift(L).fillna(0.0)

    total_shifted = df['TOTAL'].shift(1)
    df['tot_mean_1h']  = total_shifted.rolling(steps_1h,  min_periods=steps_1h//2).mean()
    df['tot_std_1h']   = total_shifted.rolling(steps_1h,  min_periods=steps_1h//2).std()
    df['tot_mean_6h']  = total_shifted.rolling(steps_6h,  min_periods=steps_6h//2).mean()
    df['tot_std_6h']   = total_shifted.rolling(steps_6h,  min_periods=steps_6h//2).std()
    df['tot_mean_24h'] = total_shifted.rolling(steps_24h, min_periods=steps_24h//2).mean()
    df['tot_std_24h']  = total_shifted.rolling(steps_24h, min_periods=steps_24h//2).std()

    def safe_div(num, den, eps=1e-6):
        den = den.copy()
        den = den.where(den.abs() > eps, eps)
        return num / den

    m1_num = df['TOTAL'].shift(1)
    m1_den = df['TOTAL'].shift(1 + steps_1h)
    m6_den = df['TOTAL'].shift(1 + steps_6h)

    df['mom_1h'] = safe_div(m1_num, m1_den) - 1.0
    df['mom_6h'] = safe_div(m1_num, m6_den) - 1.0
    df['mom_1h'] = df['mom_1h'].clip(-3, 3)
    df['mom_6h'] = df['mom_6h'].clip(-3, 3)

    cal = make_calendar_features(df.index)
    exo_cols: List[str] = []
    if 'Temperature' in df.columns: exo_cols.append('Temperature')
    if 'Outage' in df.columns:      exo_cols.append('Outage')

    policy_cols = ['Holiday', 'ECQ', 'GCQ', 'ALERT LEVEL 1', 'ALERT LEVEL 2', 'ALERT LEVEL 3']
    weekday_cols = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    for col in policy_cols + weekday_cols:
        if col in df.columns:
            exo_cols.append(col)

    interaction_cols = []
    if 'Temperature' in df.columns:
        df['temp_x_drop1'] = df['Temperature'] * df['sudden_drop_lag1']
        df['temp_x_mom1h'] = df['Temperature'] * df['mom_1h']
        interaction_cols += ['temp_x_drop1', 'temp_x_mom1h']

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
    # (End of your feature engineering block)
    # -------------------------------------------------------------------------

    # ----------------------------
    # Build categorical array (weekday + optional binary flags) to use embeddings
    # ----------------------------
    # Decide which categorical features to embed (you can add more)
    cat_cols_available = []
    # Weekday from the timestamp (0=Mon .. 6=Sun)
    cat_cols_available.append('weekday')
    # binary flags if present
    if 'Outage' in df.columns:
        cat_cols_available.append('Outage')
    if 'Holiday' in df.columns:
        cat_cols_available.append('Holiday')

    # Drop possible one-hot weekday / policy columns from X_exo if they exist (we will embed instead)
    weekday_onehots = weekday_cols
    policy_onehots = policy_cols
    drop_cols = [c for c in (weekday_onehots + policy_onehots) if c in X_exo.columns]
    if drop_cols:
        X_exo = X_exo.drop(columns=drop_cols, errors='ignore')

    # Finite-only sanitization BEFORE scaling
    X_exo.replace([np.inf, -np.inf], np.nan, inplace=True)
    finite_rows = np.isfinite(X_exo.to_numpy()).all(axis=1)
    finite_target = np.isfinite(df['TOTAL'].to_numpy())
    valid_mask = finite_rows & finite_target

    df    = df.loc[valid_mask].copy()
    X_exo = X_exo.loc[valid_mask].copy()

    # Window end timestamps (each sample's last target time) AFTER filtering
    times = df.index[W + H - 1 :]

    # Splits
    train_end = pd.Timestamp('2023-12-25 23:55:00')
    val_end   = pd.Timestamp('2024-11-25 23:55:00')

    # SCALING (leak-free)
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

    # Build categorical arrays aligned with X_full.index
    cat_list = []
    # weekday from timestamp (0..6)
    weekday_int = df.index.dayofweek.to_numpy().astype(int)  # 0=Mon ... 6=Sun
    cat_list.append(weekday_int)
    if 'Outage' in df.columns:
        cat_list.append(df['Outage'].astype(int).to_numpy())
    if 'Holiday' in df.columns:
        cat_list.append(df['Holiday'].astype(int).to_numpy())

    cat_array = np.vstack(cat_list).T  # [N, C]
    # For safety, ensure non-negative integer dtype
    cat_array = cat_array.astype(int)

    # Final continuous data (exclude target)
    cont_array = X_feats_scaled  # [N, F_cont]
    y_array = y_scaled           # [N, 1]
    n_cont_features = cont_array.shape[1]

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
    full_ds = LoadWindowDataset(cont_array, cat_array, y_array, W, H)
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
    # Determine vocab sizes for categorical inputs
    cat_vocab_sizes = []
    # weekday vocab = 7
    cat_vocab_sizes.append(7)
    if 'Outage' in df.columns:
        cat_vocab_sizes.append(int(df['Outage'].max()) + 1)  # expect 0/1
    if 'Holiday' in df.columns:
        cat_vocab_sizes.append(int(df['Holiday'].max()) + 1)

    model = LSTM_Forecaster(n_cont_features=n_cont_features,
                           cat_vocab_sizes=cat_vocab_sizes,
                           window=W, horizon=H,
                           enc1=256, enc2=128, latent_dim=64, p_drop=0.2
                           ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
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
    # Training Loop (unpack cont + cat)
    # ----------------------------
    for ep in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        tl = 0.0
        for Xb_cont, Xb_cat, Yseqb in train_loader:
            Xb_cont, Xb_cat, Yseqb = Xb_cont.to(device), Xb_cat.to(device), Yseqb.to(device)
            yhat = model(Xb_cont, Xb_cat)
            loss = f_fn(yhat, Yseqb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * Xb_cont.size(0)
        train_losses.append(tl / len(train_ds))

        # Validation
        model.eval()
        vl, preds_last_scaled, truths_last_scaled = 0.0, [], []
        preds_full_scaled, truths_full_scaled = [], []
        with torch.no_grad():
            for Xb_cont, Xb_cat, Yseqb in val_loader:
                Xb_cont, Xb_cat, Yseqb = Xb_cont.to(device), Xb_cat.to(device), Yseqb.to(device)
                yhat = model(Xb_cont, Xb_cat)
                vl += f_fn(yhat, Yseqb).item() * Xb_cont.size(0)
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
        mape = smape(inv_t, inv_p)

        pf = np.concatenate(preds_full_scaled, axis=0)
        tf = np.concatenate(truths_full_scaled, axis=0)
        inv_pf = tgt_scaler.inverse_transform(pf.reshape(-1,1))[:,0]
        inv_tf = tgt_scaler.inverse_transform(tf.reshape(-1,1))[:,0]
        rm_all  = root_mean_squared_error(inv_tf, inv_pf)
        mae_all = mean_absolute_error(inv_tf, inv_pf)
        mape_all = smape(inv_tf, inv_pf)


        cur_lr = opt.param_groups[0]['lr']
        print(f"Ep{ep:02d} Train={train_losses[-1]:.4f} | Val={val_loss:.4f} | "
                f"Val(last) RMSE={rm:.1f} MAE={mae:.1f} MAPE={mape:.2f}% R²={r2:.3f} | "
                f"Val(full) RMSE={rm_all:.1f} MAE={mae_all:.1f} MAPE={mape_all:.2f}% | "
                f"LR={cur_lr:.6f} | Time={time.time()-t0:.1f}s")


        _ = es.step(val_loss, model=model)
        if es.should_stop:
            if es.best_state is not None:
                model.load_state_dict(es.best_state)
                if es.verbose:
                    print("EarlyStopping: best weights restored.")
            break

    # ----------------------------
    # (rest of script: plotting/testing) -- unchanged except DataLoader unpacking
    # ----------------------------
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Weighted MSE'); plt.legend(); plt.title('Loss Curves (tail-weighted)')
    plt.tight_layout(); plt.show()

    # Testing
    model.eval()
    preds_last_scaled, truths_last_scaled = [], []
    preds_full_scaled, truths_full_scaled = [], []
    test_vl = 0.0
    with torch.no_grad():
        for Xb_cont, Xb_cat, Yseqb in test_loader:
            Xb_cont, Xb_cat, Yseqb = Xb_cont.to(device), Xb_cat.to(device), Yseqb.to(device)
            yhat = model(Xb_cont, Xb_cat)
            test_vl += f_fn(yhat, Yseqb).item() * Xb_cont.size(0)
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
    n, p = len(inv_t), (n_cont_features - 1)  # p ~ number of non-target features (approx)
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

    # (Plots similar to your original code)
    plt.figure(figsize=(10,4))
    plt.plot(times[test_idx], inv_t, label='Actual')
    plt.plot(times[test_idx], inv_p, label='Predicted', alpha=0.7)
    plt.title("24h-Ahead Forecast – Last-Step (Test)")
    plt.xlabel("Timestamp"); plt.ylabel("TOTAL demand")
    plt.legend(); plt.tight_layout(); plt.show()

    # show a few sample trajectories
    num_show = min(3, len(test_idx))
    if num_show > 0:
        # get a small batch from test_ds
        Xb_cont, Xb_cat, Yseqb = next(iter(DataLoader(test_ds, batch_size=num_show, shuffle=False)))
        Xb_cont = Xb_cont.to(device); Xb_cat = Xb_cat.to(device)
        with torch.no_grad():
            yhat = model(Xb_cont, Xb_cat).cpu().numpy()
        ytrue = Yseqb.numpy()
        for i in range(min(num_show, Xb_cont.size(0))):
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
