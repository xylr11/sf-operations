import polars as pl
import numpy as np
import sf_quant.data as sfd
import datetime as dt
from sf_quant.data._factors import factors
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset

class IdioVolDatasetBarra(Dataset):
    """
    Polars-only dataset for forecasting realized idiosyncratic volatility
    using Barra factor exposures and Barra specific returns.
    """

    def __init__(
        self,
        exposure_df: pl.DataFrame,
        spec_ret_df: pl.DataFrame,
        feature_cols,
        forward_window=21,
        time_decay_half_life: int | None = 126,
    ):
        self.k = forward_window
        self.feature_cols = feature_cols

        # Merge exposures (features) with Barra specific returns
        df = (
            exposure_df
            .join(spec_ret_df, on=["barrid", "date"], how="inner")
            .sort(["barrid", "date"])
        )

        # add age in days for time-based weighting
        age_days = (
            df
            .with_columns((pl.max("date") - pl.col("date")).dt.total_days().alias("age_days"))
            .select("age_days")
            .to_numpy()
            .reshape(-1)
            .astype(np.float32)
        )

        if time_decay_half_life is not None and time_decay_half_life > 0:
            decay_rate = np.log(2.0) / float(time_decay_half_life)
            weights = np.exp(-decay_rate * age_days).astype(np.float32)
        else:
            weights = np.ones_like(age_days, dtype=np.float32)

        self.df = df
        self.sample_weights = weights

        # Convert to numpy for fast row-indexing during training
        self.feature_mat = df.select(feature_cols).to_numpy()
        self.spec_ret = df["specific_return"].to_numpy()

        # Precompute all valid (t, window) index pairs
        self.index_tuples, self.prev_window_indices = self._build_index()

    def _build_index(self):
        idx_list = []
        prev_windows = []

        # group by asset to ensure consecutive rows belong to the same name
        for _, g in self.df.group_by("barrid", maintain_order=True):
            # absolute row indices in original df
            phys = g.select(pl.row_index())["index"].to_list()

            # build windows
            for i in range(len(phys) - self.k):
                t_idx = phys[i]
                win_idx = phys[i : i + self.k]  # future window
                idx_list.append((t_idx, win_idx))
                prev_windows.append(phys[i - 1 : i - 1 + self.k] if i > 0 else None)

        return idx_list, prev_windows

    def __len__(self):
        return len(self.index_tuples)

    def __getitem__(self, i):
        t_idx, win_idx = self.index_tuples[i]

        # input features at time t
        x = self.feature_mat[t_idx].astype(np.float32)

        # realized specific volatility over future window
        eps = self.spec_ret[win_idx]
        realized_var = 252. * np.mean(eps ** 2)
        y = np.log(realized_var + 1e-12).astype(np.float32)

        w = self.sample_weights[t_idx].astype(np.float32)

        return torch.tensor(x), torch.tensor(y), torch.tensor(w)

def get_idio_vol_dataloaders(
    start,
    end,
    forward_window=63,
    debug=False,
    val_proportion=0.2,
    batch_size=1024,
    num_workers=4,
    time_split=True,
    gap_days=None,
    time_decay_half_life: int | None = 126,
):
    if debug: print(f"[INFO] Started loading data from {start} to {end}")
    exposures = (
        sfd.load_exposures(start, end, True, ["date", "barrid"] + factors)
        .fill_nan(0)
        .fill_null(0)
    )
    if debug: print(f"[INFO] Loaded exposures from {start} to {end}")
    specific_returns = (
        sfd.load_assets(start, end, ["date", "barrid", "specific_return"], in_universe=True)
        .fill_nan(0)
        .fill_null(0)
        .with_columns(pl.col("specific_return") / 100)
    )
    
    if debug: print(f"[INFO] Loaded specific returns from {start} to {end}")
    # time-based split to avoid leakage across overlapping windows
    if time_split and 0 < val_proportion < 1:
        unique_dates = sorted(
            exposures.select(pl.col("date").unique()).to_series().to_list()
        )
        if len(unique_dates) < 2:
            if debug: print("[WARN] Not enough dates for time split; falling back to random split.")
            time_split = False
        else:
            cutoff_idx = int(len(unique_dates) * (1 - val_proportion))
            cutoff_idx = max(1, min(cutoff_idx, len(unique_dates) - 1))
            gap = gap_days if gap_days is not None else forward_window
            val_start_idx = min(cutoff_idx + gap, len(unique_dates) - 1)
            cutoff_date = unique_dates[cutoff_idx]
            val_start_date = unique_dates[val_start_idx]
            if debug:
                print(f"[INFO] Using cutoff date {cutoff_date} for time-based split with {gap} day gap; val starts {val_start_date}.")

            train_exposures = exposures.filter(pl.col("date") < cutoff_date)
            val_exposures = exposures.filter(pl.col("date") >= val_start_date)
            train_spec = specific_returns.filter(pl.col("date") < cutoff_date)
            val_spec = specific_returns.filter(pl.col("date") >= val_start_date)

            train_ds = IdioVolDatasetBarra(
                exposure_df=train_exposures,
                spec_ret_df=train_spec,
                feature_cols=factors,
                forward_window=forward_window,
                time_decay_half_life=time_decay_half_life,
            )
            val_ds = IdioVolDatasetBarra(
                exposure_df=val_exposures,
                spec_ret_df=val_spec,
                feature_cols=factors,
                forward_window=forward_window,
                time_decay_half_life=time_decay_half_life,
            )
    if not time_split or not (0 < val_proportion < 1):
        dataset = IdioVolDatasetBarra(
            exposure_df=exposures,
            spec_ret_df=specific_returns,
            feature_cols=factors,
            forward_window=forward_window,
            time_decay_half_life=time_decay_half_life,
        )
        if debug: print(f'[INFO] Finished making dataset from {start} to {end}')
        n_total = len(dataset)
        if n_total == 0:
            if debug: print("[WARN] Dataset is empty; returning empty loaders.")
            empty_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            return empty_loader, empty_loader

        val_size = int(n_total * val_proportion)
        if val_proportion > 0 and val_size == 0:
            val_size = 1  # ensure non-empty val set when requested
        if val_size >= n_total:
            val_size = n_total - 1  # keep at least one train sample
        train_size = n_total - val_size

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader

def evaluate_barra_specific_risk(start, end, forward_window=63, debug=False):
    """
    Align Barra specific risk forecasts with realized idiosyncratic variance and
    return the evaluation frame plus Barra's log-variance MSE.
    """
    if debug: print(f"[INFO] Loading assets from {start} to {end}")
    assets = (
        sfd.load_assets(
            start,
            end,
            ["date", "barrid", "specific_return", "specific_risk"],
            in_universe=True,
        )
        .fill_nan(0)
        .fill_null(0)
        .with_columns(pl.col("specific_return") / 100)
        .sort(["barrid", "date"])
    )

    rows = []
    for _, g in assets.group_by("barrid", maintain_order=True):
        idx = g.select(pl.row_index())["index"].to_list()
        eps = g["specific_return"].to_numpy()
        sigma = g["specific_risk"].to_numpy()
        dates = g["date"].to_list()
        name = g["barrid"][0]

        for i in range(len(idx) - forward_window):
            window_eps = eps[i : i + forward_window]
            realized_var = 252. * float(np.mean(np.square(window_eps)))
            barra_var = float((sigma[i] ** 2) / 1e4)
            rows.append(
                {
                    "date": dates[i],
                    "barrid": name,
                    "realized_var": realized_var,
                    "barra_var": barra_var,
                }
            )

    eval_df = pl.DataFrame(rows)
    if eval_df.is_empty():
        if debug: print("[WARN] Evaluation frame is empty.")
        return eval_df, np.nan

    eps = 1e-12  # numerical stability for log
    eval_df = eval_df.with_columns(
        [
            (pl.col("realized_var") + eps).log().alias("realized_logvar"),
            (pl.col("barra_var") + eps).log().alias("barra_logvar"),
        ]
    )

    mse_barra = float(
        ((eval_df["barra_logvar"] - eval_df["realized_logvar"]) ** 2).mean()
    )
    if debug: print(f"[INFO] Barra log-var MSE: {mse_barra:.6f}")
    return eval_df, mse_barra

class IdioVolMLP(nn.Module):
    """
    Simple MLP that maps factor exposures to a forecast of realized log
    idiosyncratic volatility.
    """

    def __init__(
        self,
        input_dim: int = len(factors),
        hidden_sizes=(512, 128, 32),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        # final layer outputs a single log-variance prediction
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # allow single-sample inference
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = self.network(x.float())
        return out.squeeze(-1)

def train_model(
    model: nn.Module,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_interval: int = 100,
    epochs: int = 1,
    device: torch.device | None = None,
    target_mean: float | None = None,
    target_std: float | None = None,
):
    """
    Train a model on `train_loader`, validating every `val_interval` steps.

    Returns
    -------
    dict
        History dict with lists for train/val loss.
    """
    if device is None:
        device = next(model.parameters()).device

    history = {"train_loss": [], "val_loss": []}
    global_step = 0

    for i in range(epochs):
        train_losses_epoch = []
        model.train()
        for batch in tqdm(train_loader, desc=f'Training: Epoch {i + 1}'):
            if len(batch) == 3:
                xb, yb, wb = batch
            else:
                xb, yb = batch
                wb = None

            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device) if wb is not None else None

            optimizer.zero_grad()
            preds = model(xb)
            if target_mean is not None and target_std is not None:
                preds_loss = (preds - target_mean) / target_std
                yb_loss = (yb - target_mean) / target_std
            else:
                preds_loss = preds
                yb_loss = yb

            if wb is not None:
                loss_vec = (preds_loss - yb_loss) ** 2
                loss = (loss_vec * wb).sum() / (wb.sum() + 1e-12)
            else:
                loss = loss_fn(preds_loss, yb_loss)

            loss.backward()
            optimizer.step()

            history["train_loss"].append((global_step, float(loss.detach().cpu())))
            train_losses_epoch.append(float(loss.detach().cpu()))

            global_step += 1

            # run validation every `val_interval` steps
            if val_interval and global_step % val_interval == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        if len(val_batch) == 3:
                            vx, vy, vw = val_batch
                        else:
                            vx, vy = val_batch
                            vw = None

                        vx = vx.to(device)
                        vy = vy.to(device)
                        vw = vw.to(device) if vw is not None else None

                        vpred = model(vx)
                        if target_mean is not None and target_std is not None:
                            vpred_loss = (vpred - target_mean) / target_std
                            vy_loss = (vy - target_mean) / target_std
                        else:
                            vpred_loss = vpred
                            vy_loss = vy

                        if vw is not None:
                            vloss_vec = (vpred_loss - vy_loss) ** 2
                            vloss = (vloss_vec * vw).sum() / (vw.sum() + 1e-12)
                        else:
                            vloss = loss_fn(vpred_loss, vy_loss)
                        val_losses.append(float(vloss.detach().cpu()))

                if val_losses:
                    history["val_loss"].append((global_step, float(np.mean(val_losses))))

                model.train()

        # end-of-epoch validation/logging
        val_losses_epoch = []
        val_preds_all = []
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                if len(val_batch) == 3:
                    vx, vy, vw = val_batch
                else:
                    vx, vy = val_batch
                    vw = None

                vx = vx.to(device)
                vy = vy.to(device)
                vw = vw.to(device) if vw is not None else None

                vpred = model(vx)
                if target_mean is not None and target_std is not None:
                    vpred_loss = (vpred - target_mean) / target_std
                    vy_loss = (vy - target_mean) / target_std
                else:
                    vpred_loss = vpred
                    vy_loss = vy

                if vw is not None:
                    vloss_vec = (vpred_loss - vy_loss) ** 2
                    vloss = (vloss_vec * vw).sum() / (vw.sum() + 1e-12)
                else:
                    vloss = loss_fn(vpred_loss, vy_loss)

                val_losses_epoch.append(float(vloss.detach().cpu()))
                val_preds_all.append(vpred.detach().cpu())
        model.train()

        epoch_train_mean = float(np.mean(train_losses_epoch)) if train_losses_epoch else float("nan")
        epoch_val_mean = float(np.mean(val_losses_epoch)) if val_losses_epoch else float("nan")

        if val_preds_all:
            preds_cat = torch.cat(val_preds_all)
            pred_mean = float(preds_cat.mean())
            pred_var = float(preds_cat.var(unbiased=False))
            print(
                f"Epoch {i + 1}/{epochs} - train_loss: {epoch_train_mean:.6f}, "
                f"val_loss: {epoch_val_mean:.6f}, pred_mean: {pred_mean:.6f}, pred_var: {pred_var:.6f}"
            )
        else:
            print(f"Epoch {i + 1}/{epochs} - train_loss: {epoch_train_mean:.6f}, val_loss: {epoch_val_mean:.6f}")

    return history

def constant_baseline_loss(train_loader: DataLoader, val_loader: DataLoader, loss_fn, device):
    """
    Compute validation loss for a constant predictor equal to the train-set mean target.
    """
    with torch.no_grad():
        ys = []
        for batch in train_loader:
            if len(batch) == 3:
                _, yb, _ = batch
            else:
                _, yb = batch
            ys.append(yb)
        if not ys:
            return None
        y_train = torch.cat(ys).to(device)

        val_losses = []
        for batch in val_loader:
            if len(batch) == 3:
                _, vy, vw = batch
            else:
                _, vy = batch
                vw = None

            vy = vy.to(device)
            vw = vw.to(device) if vw is not None else None
            # broadcast baseline mean to val shape
            const_pred = torch.full_like(vy, y_train.mean())
            if vw is not None:
                loss_vec = (const_pred - vy) ** 2
                val_loss = (loss_vec * vw).sum() / (vw.sum() + 1e-12)
            else:
                val_loss = loss_fn(const_pred, vy)
            val_losses.append(float(val_loss.detach().cpu()))

    return float(np.mean(val_losses)) if val_losses else None

def _unwrap_base_dataset(loader: DataLoader):
    """
    Return the underlying dataset and indices for both Dataset and Subset wrappers.
    """
    ds = loader.dataset
    if isinstance(ds, Subset):
        return ds.dataset, ds.indices
    return ds, range(len(ds))

def last_period_baseline_loss(val_loader: DataLoader, loss_fn, device, eps: float = 1e-12):
    """
    Compute validation loss for a baseline that predicts the prior period's realized
    idiosyncratic volatility (log variance) for each sample.
    """
    _ = loss_fn  # placeholder to mirror signature of other baselines
    base_ds, indices = _unwrap_base_dataset(val_loader)
    if not isinstance(base_ds, IdioVolDatasetBarra):
        return None

    prev_windows = getattr(base_ds, "prev_window_indices", None)
    if prev_windows is None:
        return None

    val_losses = []
    with torch.no_grad():
        for ds_idx in indices:
            prev_win = prev_windows[ds_idx]
            if prev_win is None:
                continue  # no prior window available for this asset

            sample = base_ds[ds_idx]
            if len(sample) == 3:
                _, yb, wb = sample
            else:
                _, yb = sample
                wb = None

            prev_eps = base_ds.spec_ret[prev_win]
            prev_var = 252. * float(np.mean(np.square(prev_eps)))
            pred_val = np.log(prev_var + eps)

            pred = torch.tensor(pred_val, device=device, dtype=yb.dtype)
            yb = yb.to(device)
            wb = wb.to(device) if wb is not None else None

            diff_sq = (pred - yb) ** 2
            if wb is not None:
                weight_sum = wb.sum() if wb.ndim > 0 else wb
                val_loss = (diff_sq * wb).sum() / (weight_sum + 1e-12)
            else:
                val_loss = diff_sq.mean()

            val_losses.append(float(val_loss.detach().cpu()))

    return float(np.mean(val_losses)) if val_losses else None

def target_stats(
    loader: DataLoader,
    plot: bool = False,
    plot_path: str | None = None,
    show: bool = False,
):
    """
    Aggregate target statistics and optionally plot distribution and decile variance
    to visually check stability (homoscedasticity) of log targets.
    """
    ys = []
    for batch in loader:
        if len(batch) == 3:
            _, yb, _ = batch
        else:
            _, yb = batch
        ys.append(yb)
    if not ys:
        return None
    y_cat = torch.cat(ys)

    stats = {
        "mean": float(y_cat.mean()),
        "var": float(y_cat.var(unbiased=False)),
        "std": float(y_cat.std(unbiased=False)),
    }

    if not plot:
        return stats

    y_np = y_cat.cpu().numpy()
    mu = stats["mean"]
    sigma = stats["std"]
    sigma_safe = sigma if sigma > 0 else 1e-8

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Distribution of log variances with normal overlay
    ax0 = axes[0]
    ax0.hist(y_np, bins=50, density=True, alpha=0.6, color="steelblue")
    grid = np.linspace(y_np.min(), y_np.max(), 200)
    normal_pdf = (1 / (sigma_safe * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((grid - mu) / sigma_safe) ** 2)
    ax0.plot(grid, normal_pdf, color="darkorange", label="Normal PDF (mu/std)")
    ax0.axvline(mu, color="black", linestyle="--", linewidth=1, label="Mean")
    ax0.set_title("Target log-var distribution")
    ax0.legend()

    # Variance per target decile as a quick homoscedasticity check
    ax1 = axes[1]
    deciles = np.linspace(0, 1, 11)
    q = np.quantile(y_np, deciles)
    bucket_ids = np.digitize(y_np, q[1:-1], right=True)
    bucket_vars = []
    for i in range(10):
        mask = bucket_ids == i
        if mask.any():
            bucket_vars.append(np.var(y_np[mask]))
        else:
            bucket_vars.append(np.nan)
    ax1.bar(range(1, 11), bucket_vars, color="seagreen")
    ax1.set_xlabel("Target decile")
    ax1.set_ylabel("Variance")
    ax1.set_title("Decile-wise target variance")

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return stats

def volatile_training_example():
    start = dt.date(2018, 1, 1)
    end = dt.date(2021, 12, 31)

    device = torch.device('cpu') # Change this eventually, but will need to submit slurm scripts to be granted gpu access

    # build train/val loaders
    train_loader, val_loader = get_idio_vol_dataloaders(
        start,
        end,
        forward_window=21,
        debug=True,
        val_proportion=0.3,
        batch_size=512,
        num_workers=4,
        gap_days=21,
        time_decay_half_life=42
    )

    train_stats = target_stats(train_loader, plot=True, plot_path="train_targets.png")
    val_stats = target_stats(val_loader, plot=True, plot_path="val_targets.png")
    print(f"Train target stats: {train_stats}")
    print(f"Val target stats:   {val_stats}")

    model = IdioVolMLP().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    history = train_model(
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        val_interval=200,
        epochs=6,
        device=device,
        target_mean=None, # Removed
        target_std=None, # Removed
    )

    # simple summary of last recorded losses
    if history["train_loss"]:
        print(f"Final train loss @ step {history['train_loss'][-1][0]}: {history['train_loss'][-1][1]:.6f}")
    if history["val_loss"]:
        print(f"Final val loss @ step {history['val_loss'][-1][0]}: {history['val_loss'][-1][1]:.6f}")

    # baseline comparison: prior-period realized idio vol
    last_period_val_loss = last_period_baseline_loss(val_loader, loss_fn, device)
    if last_period_val_loss is not None:
        print(f"Last-period baseline val loss: {last_period_val_loss:.6f}")

    # baseline comparison: constant predictor at train mean target
    baseline_val_loss = constant_baseline_loss(train_loader, val_loader, loss_fn, device)
    if baseline_val_loss is not None:
        print(f"Constant baseline val loss: {baseline_val_loss:.6f}")

def calm_training_example():
    start = dt.date(2012, 1, 1)
    end = dt.date(2014, 12, 31)

    device = torch.device('cpu') # Change this eventually, but will need to submit slurm scripts to be granted gpu access

    # build train/val loaders
    train_loader, val_loader = get_idio_vol_dataloaders(
        start,
        end,
        forward_window=63,
        debug=True,
        val_proportion=0.3,
        batch_size=512,
        num_workers=4,
        gap_days=63,
        time_decay_half_life=126
    )

    train_stats = target_stats(train_loader, plot=True, plot_path="train_targets.png")
    val_stats = target_stats(val_loader, plot=True, plot_path="val_targets.png")
    print(f"Train target stats: {train_stats}")
    print(f"Val target stats:   {val_stats}")

    model = IdioVolMLP().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    history = train_model(
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        val_interval=200,
        epochs=6,
        device=device,
        target_mean=None, # Removed
        target_std=None, # Removed
    )

    # simple summary of last recorded losses
    if history["train_loss"]:
        print(f"Final train loss @ step {history['train_loss'][-1][0]}: {history['train_loss'][-1][1]:.6f}")
    if history["val_loss"]:
        print(f"Final val loss @ step {history['val_loss'][-1][0]}: {history['val_loss'][-1][1]:.6f}")

    # baseline comparison: prior-period realized idio vol
    last_period_val_loss = last_period_baseline_loss(val_loader, loss_fn, device)
    if last_period_val_loss is not None:
        print(f"Last-period baseline val loss: {last_period_val_loss:.6f}")

    # baseline comparison: constant predictor at train mean target
    baseline_val_loss = constant_baseline_loss(train_loader, val_loader, loss_fn, device)
    if baseline_val_loss is not None:
        print(f"Constant baseline val loss: {baseline_val_loss:.6f}")    

def calm_stats_example():
    start = dt.date(2012, 1, 1)
    end = dt.date(2014, 12, 31)

    # build train/val loaders
    train_loader, _ = get_idio_vol_dataloaders(
        start,
        end,
        forward_window=63,
        debug=True,
        val_proportion=0.3,
        batch_size=512,
        num_workers=4,
        gap_days=63,
        time_decay_half_life=126
    )

    target_stats(train_loader, True, 'idio_vol_model/calm_target_plot.png')

def volatile_stats_example():
    start = dt.date(2018, 1, 1)
    end = dt.date(2021, 12, 31)

    # build train/val loaders
    train_loader, _ = get_idio_vol_dataloaders(
        start,
        end,
        forward_window=21,
        debug=True,
        val_proportion=0.3,
        batch_size=512,
        num_workers=4,
        gap_days=21,
        time_decay_half_life=42
    )

    target_stats(train_loader, True, 'idio_vol_model/volatile_target_plot.png') 

if __name__ == "__main__":
    volatile_training_example()
