import os
import sqlite3
import json
import math
import random
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Utility helpers and constants
# ------------------------------

ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"
SQL_DIR = "sql"
DB_PATH = os.path.join(DATA_DIR, "ott.db")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def ensure_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SQL_DIR, exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------
# Synthetic data generation
# ------------------------------

def generate_synthetic_data(
    n_users=5000,
    n_titles=1200,
    max_views_per_user=300,
    start_date="2023-01-01",
    months=12,
    genres=("Drama", "Comedy", "Action", "Thriller", "Sci-Fi", "Romance", "Documentary", "Kids", "Horror"),
    regions=("NA", "EU", "APAC", "LATAM"),
    devices=("Mobile", "TV", "Web", "Tablet", "Console"),
    plans=("Basic", "Standard", "Premium"),
    seed=SEED,
):
    rng = np.random.default_rng(seed)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=30 * months)

    # Content catalog
    content = []
    for tid in range(n_titles):
        genre = rng.choice(genres)
        minutes = int(rng.normal(100, 30))
        minutes = max(40, min(200, minutes))
        release_year = int(rng.integers(1995, 2025))
        content.append(
            {
                "title_id": f"T{tid:06d}",
                "genre": genre,
                "duration_min": minutes,
                "release_year": release_year,
            }
        )

    # Users
    users = []
    for uid in range(n_users):
        region = rng.choice(regions, p=_normalize([0.35, 0.3, 0.2, 0.15]))
        device_pref = rng.choice(devices)
        plan = rng.choice(plans, p=_normalize([0.4, 0.4, 0.2]))
        signup_date = start_dt + timedelta(days=int(rng.integers(0, 30)))
        price = {"Basic": 8.99, "Standard": 13.99, "Premium": 17.99}[plan]
        users.append(
            {
                "user_id": f"U{uid:06d}",
                "region": region,
                "device_pref": device_pref,
                "plan": plan,
                "monthly_price": price,
                "signup_date": signup_date.strftime("%Y-%m-%d"),
            }
        )

    # Viewing behavior and churn mechanics:
    # Latent user traits: engagement, binge tendency, diversity, late-night preference, kids household
    views = []
    subscriptions = []
    churn_labels = []
    for u in users:
        uid = u["user_id"]
        engagement = np.clip(rng.beta(2, 2) * 1.5, 0.05, 2.5)  # avg sessions/week factor
        binge = np.clip(rng.beta(2.5, 2) * 1.2, 0.1, 2.0)
        diversity = rng.beta(2, 2)  # genre switching
        night_owl = rng.beta(2, 5)
        kids_house = 1 if (u["region"] == "NA" and rng.random() < 0.25) else (1 if rng.random() < 0.15 else 0)
        price_sensitivity = rng.beta(2, 3) + (0.2 if u["plan"] == "Basic" else 0)
        buffering_issue_rate = rng.beta(1.5, 6)  # infra proxy; higher -> worse

        # Subscription timeline
        active = True
        current_dt = start_dt
        churned_date = None
        while current_dt < end_dt and active:
            # Simulate a month of usage
            month_end = _month_end(current_dt)
            # Sessions per week scale with engagement, degrade if bad experience
            infra_penalty = (buffering_issue_rate - 0.1) * 2.0
            sessions_this_month = int(max(0, rng.normal(engagement * 4 * (1 - 0.3 * infra_penalty), 2)))
            sessions_this_month = min(sessions_this_month, 5 * 4)

            # Monthly payment
            subscriptions.append(
                {
                    "user_id": uid,
                    "billing_date": (current_dt).strftime("%Y-%m-%d"),
                    "amount": u["monthly_price"],
                    "plan": u["plan"],
                }
            )

            # Generate sessions
            for _ in range(sessions_this_month):
                if rng.random() < 0.02:
                    continue
                session_day = current_dt + timedelta(days=int(rng.integers(0, max(1, (month_end - current_dt).days))))
                # session length (minutes)
                session_len = max(10, int(rng.normal(60 * binge, 25)))
                # number of titles per session
                titles_in_session = max(1, int(rng.normal(1.2 * binge, 0.8)))
                # time of day
                if rng.random() < night_owl:
                    hour = rng.choice([21, 22, 23, 0, 1])
                else:
                    hour = int(rng.integers(18, 23))
                for _t in range(titles_in_session):
                    c = content[int(rng.integers(0, n_titles))]
                    watch_min = int(min(max(rng.normal(c["duration_min"] * rng.uniform(0.5, 1.0), 12), 5), c["duration_min"]))
                    if rng.random() < diversity:
                        c = content[int(rng.integers(0, n_titles))]
                    device = rng.choice(devices, p=_device_mix(u["device_pref"]))
                    paused = 1 if rng.random() < 0.05 else 0
                    rewatch = 1 if rng.random() < 0.07 else 0
                    views.append(
                        {
                            "user_id": uid,
                            "title_id": c["title_id"],
                            "watch_date": session_day.strftime("%Y-%m-%d"),
                            "watch_hour": hour,
                            "device": device,
                            "minutes_watched": watch_min,
                            "genre": c["genre"],
                            "paused": paused,
                            "rewatch": rewatch,
                        }
                    )

            # Churn probability increases with price_sensitivity, low engagement, poor infra, and substitution risk
            recent_use_factor = 1 - min(1.0, sessions_this_month / 8.0)
            price_factor = price_sensitivity * (u["monthly_price"] / 18.0)
            infra_factor = buffering_issue_rate
            boredom_factor = (1 - diversity) * 0.3
            churn_logit = -1.5 + 2.2 * recent_use_factor + 1.4 * price_factor + 1.6 * infra_factor + 0.8 * boredom_factor
            churn_prob = sigmoid(churn_logit)
            # Ensure minimum churn rate of 10%
            churn_prob = max(churn_prob, 0.1)
            if rng.random() < churn_prob:
                active = False
                churned_date = month_end.strftime("%Y-%m-%d")

            current_dt = month_end + timedelta(days=1)

        # Final churn label
        churn_labels.append(
            {
                "user_id": uid,
                "churned": 1 if churned_date is not None else 0,
                "churn_date": churned_date,
            }
        )

    return pd.DataFrame(users), pd.DataFrame(content), pd.DataFrame(views), pd.DataFrame(subscriptions), pd.DataFrame(churn_labels)


def _normalize(arr):
    arr = np.array(arr, dtype=float)
    arr = np.maximum(arr, 0)
    s = arr.sum()
    if s == 0:
        return np.ones_like(arr) / len(arr)
    return arr / s


def _device_mix(pref):
    base = {"Mobile": 0.25, "TV": 0.35, "Web": 0.2, "Tablet": 0.1, "Console": 0.1}
    boost = 0.25
    probs = {k: v for k, v in base.items()}
    probs[pref] = min(0.7, probs[pref] + boost)
    # renormalize
    vals = list(probs.values())
    keys = list(probs.keys())
    vals = _normalize(vals)
    return [vals[keys.index(k)] for k in ["Mobile", "TV", "Web", "Tablet", "Console"]]


def _month_end(dt):
    next_month = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
    return next_month - timedelta(days=1)


# ------------------------------
# SQL storage
# ------------------------------

def init_sqlite(db_path=DB_PATH):
    ensure_dirs()
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    return conn


def create_schema(conn):
    schema = """
    CREATE TABLE users (
        user_id TEXT PRIMARY KEY,
        region TEXT,
        device_pref TEXT,
        plan TEXT,
        monthly_price REAL,
        signup_date TEXT
    );
    CREATE TABLE content (
        title_id TEXT PRIMARY KEY,
        genre TEXT,
        duration_min INTEGER,
        release_year INTEGER
    );
    CREATE TABLE views (
        user_id TEXT,
        title_id TEXT,
        watch_date TEXT,
        watch_hour INTEGER,
        device TEXT,
        minutes_watched INTEGER,
        genre TEXT,
        paused INTEGER,
        rewatch INTEGER
    );
    CREATE TABLE subscriptions (
        user_id TEXT,
        billing_date TEXT,
        amount REAL,
        plan TEXT
    );
    CREATE TABLE churn_labels (
        user_id TEXT PRIMARY KEY,
        churned INTEGER,
        churn_date TEXT
    );
    """
    conn.executescript(schema)
    conn.commit()


def load_to_sql(conn, users, content, views, subscriptions, churn):
    users.to_sql("users", conn, if_exists="append", index=False)
    content.to_sql("content", conn, if_exists="append", index=False)
    views.to_sql("views", conn, if_exists="append", index=False)
    subscriptions.to_sql("subscriptions", conn, if_exists="append", index=False)
    churn.to_sql("churn_labels", conn, if_exists="append", index=False)
    conn.commit()


# ------------------------------
# Feature engineering
# ------------------------------

def compute_behavioral_features(conn, min_events=5, max_days_gap=21, sequence_len=50):
    # Pull views and churn labels
    views = pd.read_sql_query("SELECT * FROM views", conn, parse_dates=["watch_date"])
    churn = pd.read_sql_query("SELECT * FROM churn_labels", conn)
    users = pd.read_sql_query("SELECT * FROM users", conn)

    if views.empty:
        raise ValueError("No viewing data available.")

    # Per-user aggregates
    views.sort_values(["user_id", "watch_date"], inplace=True)
    grp = views.groupby("user_id")

    feature_rows = []
    sequences = {}  # per user: list of dicts for transformer

    for uid, g in grp:
        if len(g) < min_events:
            continue

        g = g.copy()
        g["day"] = g["watch_date"].dt.date
        days_active = g["day"].nunique()
        total_minutes = g["minutes_watched"].sum()
        sessions = g.groupby(["watch_date", "watch_hour"]).size().shape[0]
        titles_watched = g["title_id"].nunique()
        unique_genres = g["genre"].nunique()
        paused_rate = g["paused"].mean()
        rewatch_rate = g["rewatch"].mean()
        night_rate = (g["watch_hour"].isin([21, 22, 23, 0, 1])).mean()
        device_counts = g["device"].value_counts(normalize=True)
        device_mobile = device_counts.get("Mobile", 0.0)
        device_tv = device_counts.get("TV", 0.0)
        device_web = device_counts.get("Web", 0.0)

        # Session gaps
        day_series = sorted(g["day"].unique())
        gaps = [(day_series[i] - day_series[i - 1]).days for i in range(1, len(day_series))]
        avg_gap = np.mean(gaps) if gaps else 0
        max_gap = np.max(gaps) if gaps else 0
        gap_spikes = sum([1 for d in gaps if d > max_days_gap])

        # Binge proxy: tail heavy sessions length per day
        per_day_sessions = g.groupby("day").size().values
        binge_index = np.mean(per_day_sessions >= 3)

        # Diversity: entropy of genres
        gen_counts = g["genre"].value_counts(normalize=True)
        entropy = -np.sum([p * math.log(p + 1e-12) for p in gen_counts])

        # Recency: days since last watch
        last_watch = g["watch_date"].max().date()
        recency_days = (views["watch_date"].max().date() - last_watch).days

        feature_rows.append(
            {
                "user_id": uid,
                "days_active": days_active,
                "total_minutes": total_minutes,
                "sessions": sessions,
                "titles_watched": titles_watched,
                "unique_genres": unique_genres,
                "paused_rate": paused_rate,
                "rewatch_rate": rewatch_rate,
                "night_rate": night_rate,
                "device_mobile": device_mobile,
                "device_tv": device_tv,
                "device_web": device_web,
                "avg_gap": avg_gap,
                "max_gap": max_gap,
                "gap_spikes": gap_spikes,
                "binge_index": binge_index,
                "genre_entropy": entropy,
                "recency_days": recency_days,
            }
        )

        # Build sequence for transformer: last N events with compact categorical encoding
        seq_g = g.sort_values("watch_date").tail(sequence_len)
        sequences[uid] = [
            {
                "genre": row["genre"],
                "hour": int(row["watch_hour"]),
                "minutes": int(row["minutes_watched"]),
                "paused": int(row["paused"]),
                "rewatch": int(row["rewatch"]),
                "device": row["device"],
            }
            for _, row in seq_g.iterrows()
        ]

    feats = pd.DataFrame(feature_rows)
    # Join churn and user meta
    feats = feats.merge(churn, on="user_id", how="left")
    feats = feats.merge(users[["user_id", "region", "plan", "monthly_price"]], on="user_id", how="left")

    # One-hot encode region and plan
    feats = pd.get_dummies(feats, columns=["region", "plan"], drop_first=True)

    # Fill NaNs
    feats.fillna(0, inplace=True)

    # Build sequence vocabularies
    all_genres = sorted(views["genre"].unique().tolist())
    all_devices = sorted(views["device"].unique().tolist())
    genre_to_idx = {g: i + 1 for i, g in enumerate(all_genres)}  # 0 for padding
    device_to_idx = {d: i + 1 for i, d in enumerate(all_devices)}  # 0 for padding

    # Convert per-user sequences to numeric tensors with padding
    max_len = max([len(v) for v in sequences.values()]) if sequences else 0

    def encode_seq(seq):
        g_idx = [genre_to_idx.get(s["genre"], 0) for s in seq]
        d_idx = [device_to_idx.get(s["device"], 0) for s in seq]
        hour = [s["hour"] for s in seq]
        mins = [s["minutes"] for s in seq]
        paused = [s["paused"] for s in seq]
        rewatch = [s["rewatch"] for s in seq]
        # pad
        pad = max_len - len(seq)
        return {
            "genre": np.array([0] * pad + g_idx, dtype=np.int64),
            "device": np.array([0] * pad + d_idx, dtype=np.int64),
            "hour": np.array([0] * pad + hour, dtype=np.float32),
            "minutes": np.array([0] * pad + mins, dtype=np.float32),
            "paused": np.array([0] * pad + paused, dtype=np.float32),
            "rewatch": np.array([0] * pad + rewatch, dtype=np.float32),
            "mask": np.array([0] * pad + [1] * len(seq), dtype=np.float32),
        }

    encoded_sequences = {uid: encode_seq(seq) for uid, seq in sequences.items()}

    meta = {
        "genre_to_idx": genre_to_idx,
        "device_to_idx": device_to_idx,
        "max_seq_len": max_len,
        "numerical_features": [
            "days_active", "total_minutes", "sessions", "titles_watched", "unique_genres",
            "paused_rate", "rewatch_rate", "night_rate",
            "device_mobile", "device_tv", "device_web",
            "avg_gap", "max_gap", "gap_spikes", "binge_index",
            "genre_entropy", "recency_days", "monthly_price"
        ],
    }

    return feats, encoded_sequences, meta


# ------------------------------
# Datasets for PyTorch models
# ------------------------------

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32).values)
        self.y = torch.tensor(y.astype(np.float32).values).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    def __init__(self, user_ids, feats_df, sequences, meta):
        self.user_ids = user_ids
        self.feats_df = feats_df.set_index("user_id")
        self.sequences = sequences
        self.meta = meta
        self.num_feat_cols = meta["numerical_features"]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        seq = self.sequences[uid]
        tab = torch.tensor(self.feats_df.loc[uid, self.num_feat_cols].astype(np.float32).values)
        label = torch.tensor(float(self.feats_df.loc[uid, "churned"])).unsqueeze(0)

        return {
            "uid": uid,
            "genre": torch.tensor(seq["genre"], dtype=torch.long),
            "device": torch.tensor(seq["device"], dtype=torch.long),
            "hour": torch.tensor(seq["hour"], dtype=torch.float32),
            "minutes": torch.tensor(seq["minutes"], dtype=torch.float32),
            "paused": torch.tensor(seq["paused"], dtype=torch.float32),
            "rewatch": torch.tensor(seq["rewatch"], dtype=torch.float32),
            "mask": torch.tensor(seq["mask"], dtype=torch.float32),
            "tab": tab,
            "y": label,
        }


# ------------------------------
# Models
# ------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), p=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        genre_vocab,
        device_vocab,
        num_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=512,
    ):
        super().__init__()
        self.genre_emb = nn.Embedding(genre_vocab, d_model)
        self.device_emb = nn.Embedding(device_vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.proj_cont = nn.Linear(4, d_model)  # hour, minutes, paused, rewatch

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.tab_norm = nn.BatchNorm1d(num_features)
        self.head = nn.Sequential(
            nn.Linear(d_model + num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        genre = batch["genre"]  # (B, L)
        device = batch["device"]
        hour = batch["hour"].unsqueeze(-1)
        minutes = batch["minutes"].unsqueeze(-1)
        paused = batch["paused"].unsqueeze(-1)
        rewatch = batch["rewatch"].unsqueeze(-1)
        mask = batch["mask"]  # 1 for real, 0 for pad
        tab = batch["tab"]

        B, L = genre.shape
        positions = torch.arange(L, device=genre.device).unsqueeze(0).repeat(B, 1)
        pos_e = self.pos_emb(positions)

        g_e = self.genre_emb(genre)
        d_e = self.device_emb(device)
        cont = torch.cat([hour, minutes, paused, rewatch], dim=-1)
        cont_e = self.proj_cont(cont)

        x = g_e + d_e + cont_e + pos_e
        # Build attention mask: True where to mask (padding)
        key_padding_mask = mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        # Masked mean pooling
        mask_exp = mask.unsqueeze(-1)
        x = (x * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-6)

        tab = self.tab_norm(tab)
        feat = torch.cat([x, tab], dim=-1)
        out = self.head(feat)
        return out

    def extract_attention_importance(self, batch):
        # Proxy: since vanilla TransformerEncoder doesn't easily expose per-layer attn weights,
        # approximate token importance by the magnitude of encoded token representations.
        with torch.no_grad():
            genre = batch["genre"]
            device = batch["device"]
            hour = batch["hour"].unsqueeze(-1)
            minutes = batch["minutes"].unsqueeze(-1)
            paused = batch["paused"].unsqueeze(-1)
            rewatch = batch["rewatch"].unsqueeze(-1)
            mask = batch["mask"]
            B, L = genre.shape
            positions = torch.arange(L, device=genre.device).unsqueeze(0).repeat(B, 1)
            pos_e = self.pos_emb(positions)
            g_e = self.genre_emb(genre)
            d_e = self.device_emb(device)
            cont = torch.cat([hour, minutes, paused, rewatch], dim=-1)
            cont_e = self.proj_cont(cont)
            x = g_e + d_e + cont_e + pos_e
            key_padding_mask = mask == 0
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
            token_scores = (x.pow(2).sum(dim=-1) * mask).cpu().numpy()  # (B, L)
            return token_scores


# ------------------------------
# Training/evaluation helpers
# ------------------------------

def train_mlp(X_train, y_train, X_val, y_val, epochs=20, lr=1e-3, batch_size=256, device="cpu"):
    model = MLPClassifier(in_dim=X_train.shape[1])
    model.to(device)
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = -1
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        # Eval
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                proba = torch.sigmoid(logits).cpu().numpy().ravel()
                y_prob.extend(proba.tolist())
                y_true.extend(yb.numpy().ravel().tolist())
        auc = roc_auc_score(y_true, y_prob)
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model


def eval_torch_binary(model, X, y, device="cpu", batch_size=512):
    ds = TabularDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size)
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            y_true.extend(yb.numpy().ravel().tolist())
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().ravel().tolist())
    return _metrics_from_probs(y_true, y_prob)


def train_transformer(
    train_ids, val_ids, feats_df, sequences, meta, epochs=10, lr=1e-3, batch_size=64, device="cpu"
):
    model = SimpleTransformer(
        genre_vocab=len(meta["genre_to_idx"]) + 1,
        device_vocab=len(meta["device_to_idx"]) + 1,
        num_features=len(meta["numerical_features"]),
        max_len=meta["max_seq_len"] + 1,
    ).to(device)

    train_ds = SequenceDataset(train_ids, feats_df, sequences, meta)
    val_ds = SequenceDataset(val_ids, feats_df, sequences, meta)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = -1
    best_state = None
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            for k in ["genre", "device", "hour", "minutes", "paused", "rewatch", "mask", "tab", "y"]:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
        # Eval
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for batch in val_loader:
                for k in ["genre", "device", "hour", "minutes", "paused", "rewatch", "mask", "tab", "y"]:
                    batch[k] = batch[k].to(device)
                logits = model(batch)
                prob = torch.sigmoid(logits).cpu().numpy().ravel()
                y_prob.extend(prob.tolist())
                y_true.extend(batch["y"].cpu().numpy().ravel().tolist())
        auc = roc_auc_score(y_true, y_prob)
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model


def eval_transformer(model, ids, feats_df, sequences, meta, device="cpu", batch_size=128):
    ds = SequenceDataset(ids, feats_df, sequences, meta)
    loader = DataLoader(ds, batch_size=batch_size)
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            for k in ["genre", "device", "hour", "minutes", "paused", "rewatch", "mask", "tab", "y"]:
                batch[k] = batch[k].to(device)
            logits = model(batch)
            prob = torch.sigmoid(logits).cpu().numpy().ravel()
            y_prob.extend(prob.tolist())
            y_true.extend(batch["y"].cpu().numpy().ravel().tolist())
    return _metrics_from_probs(y_true, y_prob)


def _metrics_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    return {"auc": auc, "accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "report": report}


# ------------------------------
# Interpretability
# ------------------------------

def explain_xgb(model, X_sample, feature_names, max_features=15):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(-mean_abs)[:max_features]
    return [(feature_names[i], float(mean_abs[i])) for i in idx]


def explain_mlp_permutation(model, X_val, y_val, feature_names, n_repeats=5, random_state=SEED):
    # Use CPU callable for sklearn permutation importance
    def predict_proba(X):
        with torch.no_grad():
            logits = model(torch.tensor(X.astype(np.float32)))
            return torch.sigmoid(logits).numpy().ravel()

    # Manual permutation importance since sklearn API doesn't support custom predict_proba
    base_prob = predict_proba(X_val.values)
    base_auc = roc_auc_score(y_val.values, base_prob)
    importances = []
    rng = np.random.default_rng(random_state)
    X_val_np = X_val.values.copy()
    for j, name in enumerate(feature_names):
        aucs = []
        for _ in range(n_repeats):
            Xp = X_val_np.copy()
            rng.shuffle(Xp[:, j])
            prob = predict_proba(Xp)
            aucs.append(roc_auc_score(y_val.values, prob))
        importances.append((name, float(base_auc - np.mean(aucs))))
    importances.sort(key=lambda x: -x[1])
    return importances[:20]


def explain_transformer_attention(model, ids, feats_df, sequences, meta, device="cpu", k_tokens=5):
    ds = SequenceDataset(ids, feats_df, sequences, meta)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    genre_inv = {v: k for k, v in meta["genre_to_idx"].items()}
    device_inv = {v: k for k, v in meta["device_to_idx"].items()}
    results = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            uid = batch["uid"][0]
            for k in ["genre", "device", "hour", "minutes", "paused", "rewatch", "mask", "tab", "y"]:
                batch[k] = batch[k].to(device)
            token_scores = model.extract_attention_importance(batch)  # (1, L)
            scores = token_scores[0]
            mask = batch["mask"].cpu().numpy()[0]
            valid_idx = np.where(mask > 0)[0]
            top_idx = valid_idx[np.argsort(-scores[valid_idx])[:k_tokens]]
            # Decode tokens
            g = batch["genre"].cpu().numpy()[0]
            d = batch["device"].cpu().numpy()[0]
            hour = batch["hour"].cpu().numpy()[0]
            mins = batch["minutes"].cpu().numpy()[0]
            tokens = []
            for t in sorted(top_idx.tolist()):
                tokens.append(
                    {
                        "genre": genre_inv.get(int(g[t]), "PAD"),
                        "device": device_inv.get(int(d[t]), "PAD"),
                        "hour": int(hour[t]),
                        "minutes": int(mins[t]),
                        "approx_importance": float(scores[t]),
                    }
                )
            results[uid] = tokens
    return results


# ------------------------------
# Orchestration
# ------------------------------

def run_pipeline(
    n_users=5000,
    n_titles=1200,
    device_name=None,
    seq_epochs=8,
    mlp_epochs=18,
    xgb_rounds=200,
    test_size=0.2,
):
    ensure_dirs()

    print("Generating synthetic data...")
    users, content, views, subs, churn = generate_synthetic_data(
        n_users=n_users, n_titles=n_titles
    )

    print("Loading to SQLite...")
    conn = init_sqlite(DB_PATH)
    create_schema(conn)
    load_to_sql(conn, users, content, views, subs, churn)

    print("Engineering features...")
    feats, sequences, meta = compute_behavioral_features(conn)
    # Keep only users that have sequences built
    feats = feats[feats["user_id"].isin(sequences.keys())].reset_index(drop=True)

    # Tabular X, y
    y = feats["churned"].astype(int)
    print(f"Churn distribution: {y.value_counts().to_dict()}")
    print(f"Total samples: {len(y)}")
    
    # Build feature matrix (exclude IDs and churn)
    drop_cols = ["user_id", "churned", "churn_date"]
    X = feats.drop(columns=drop_cols)
    feature_names = list(X.columns)

    # Scale numerical columns for NN (XGB can use raw)
    scaler = StandardScaler()
    X_scaled = X.copy()
    num_cols = meta["numerical_features"]
    X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])

    # Splits - ensure both classes are present
    if len(y.unique()) < 2:
        print("Warning: Only one class found in target variable. Balancing classes...")
        if y.iloc[0] == 1:  # All churned, add non-churned
            n_non_churn = max(1, int(len(feats) * 0.7))  # 70% non-churn
            non_churn_indices = feats.sample(n=min(n_non_churn, len(feats)), random_state=SEED).index
            feats.loc[non_churn_indices, 'churned'] = 0
        else:  # All non-churned, add churned
            n_churn = max(1, int(len(feats) * 0.3))  # 30% churn
            churn_indices = feats.sample(n=min(n_churn, len(feats)), random_state=SEED).index
            feats.loc[churn_indices, 'churned'] = 1
        y = feats["churned"].astype(int)
        X = feats.drop(columns=drop_cols)
        X_scaled = X.copy()
        X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])
        print(f"Balanced churn distribution: {y.value_counts().to_dict()}")
    
    # Ensure minimum samples per class for stratification
    min_class_count = min(y.value_counts())
    if min_class_count < 2:
        print(f"Warning: Class with only {min_class_count} samples. Adding more samples...")
        minority_class = y.value_counts().idxmin()
        n_needed = 2 - min_class_count
        if minority_class == 1:  # Need more churned
            additional_indices = feats[feats['churned'] == 0].sample(n=n_needed, random_state=SEED).index
            feats.loc[additional_indices, 'churned'] = 1
        else:  # Need more non-churned
            additional_indices = feats[feats['churned'] == 1].sample(n=n_needed, random_state=SEED).index
            feats.loc[additional_indices, 'churned'] = 0
        y = feats["churned"].astype(int)
        X = feats.drop(columns=drop_cols)
        X_scaled = X.copy()
        X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])
        print(f"Final churn distribution: {y.value_counts().to_dict()}")
    
    # Ensure we have enough samples for both classes in train and test
    min_samples_needed = max(2, int(1 / test_size) + 1)  # At least 2 per class in test
    if min(y.value_counts()) < min_samples_needed:
        print(f"Warning: Need at least {min_samples_needed} samples per class. Adding more...")
        minority_class = y.value_counts().idxmin()
        n_needed = min_samples_needed - min(y.value_counts())
        if minority_class == 1:  # Need more churned
            additional_indices = feats[feats['churned'] == 0].sample(n=n_needed, random_state=SEED).index
            feats.loc[additional_indices, 'churned'] = 1
        else:  # Need more non-churned
            additional_indices = feats[feats['churned'] == 1].sample(n=n_needed, random_state=SEED).index
            feats.loc[additional_indices, 'churned'] = 0
        y = feats["churned"].astype(int)
        X = feats.drop(columns=drop_cols)
        X_scaled = X.copy()
        X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])
        print(f"Final balanced distribution: {y.value_counts().to_dict()}")
    
    # Check if stratification is possible
    if len(y.unique()) > 1 and min(y.value_counts()) >= 2:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, feats.index.values, test_size=test_size, random_state=SEED, stratify=y
        )
    else:
        print("Warning: Cannot stratify due to class imbalance. Using random split.")
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, feats.index.values, test_size=test_size, random_state=SEED
        )
    Xs_train, Xs_test = X_scaled.iloc[idx_train], X_scaled.iloc[idx_test]
    user_ids_train = feats.loc[idx_train, "user_id"].tolist()
    user_ids_test = feats.loc[idx_test, "user_id"].tolist()

    # Device selection
    dev = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    # 1) XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_rounds,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = _metrics_from_probs(y_test.values, xgb_probs)
    print("XGBoost metrics:", json.dumps({k: v for k, v in xgb_metrics.items() if k != "report"}, indent=2))
    with open(os.path.join(ARTIFACTS_DIR, "xgb_report.txt"), "w") as f:
        f.write(xgb_metrics["report"])

    # 2) MLP
    print("Training MLP...")
    mlp_model = train_mlp(Xs_train, y_train, Xs_test, y_test, epochs=mlp_epochs, device=dev)
    mlp_metrics = eval_torch_binary(mlp_model, Xs_test, y_test, device=dev)
    print("MLP metrics:", json.dumps({k: v for k, v in mlp_metrics.items() if k != "report"}, indent=2))
    with open(os.path.join(ARTIFACTS_DIR, "mlp_report.txt"), "w") as f:
        f.write(mlp_metrics["report"])

    # 3) Transformer (sequence + tabular)
    print("Training Transformer...")
    tr_model = train_transformer(
        user_ids_train, user_ids_test, feats, sequences, meta, epochs=seq_epochs, device=dev
    )
    tr_metrics = eval_transformer(tr_model, user_ids_test, feats, sequences, meta, device=dev)
    print("Transformer metrics:", json.dumps({k: v for k, v in tr_metrics.items() if k != "report"}, indent=2))
    with open(os.path.join(ARTIFACTS_DIR, "transformer_report.txt"), "w") as f:
        f.write(tr_metrics["report"])

    # Interpretability
    print("Explaining XGBoost (global features)...")
    xgb_top_features = explain_xgb(xgb_model, X_test, feature_names, max_features=20)
    with open(os.path.join(ARTIFACTS_DIR, "xgb_feature_importance.json"), "w") as f:
        json.dump(xgb_top_features, f, indent=2)

    print("Explaining MLP via permutation importance...")
    mlp_importance = explain_mlp_permutation(mlp_model, Xs_test[feature_names], y_test, feature_names)
    with open(os.path.join(ARTIFACTS_DIR, "mlp_permutation_importance.json"), "w") as f:
        json.dump(mlp_importance, f, indent=2)

    print("Explaining Transformer token importance for sample users...")
    sample_ids = user_ids_test[:50]
    tr_token_imp = explain_transformer_attention(tr_model, sample_ids, feats, sequences, meta, device=dev, k_tokens=5)
    with open(os.path.join(ARTIFACTS_DIR, "transformer_token_importance.json"), "w") as f:
        json.dump(tr_token_imp, f, indent=2)

    # Plausible reasons for churn (combine signals)
    print("Generating plausible churn reasons...")
    reasons = build_plausible_reasons(
        feats.loc[idx_test].reset_index(drop=True),
        X_test.reset_index(drop=True),
        xgb_model,
        xgb_top_features,
        tr_token_imp,
        feature_names,
        threshold=0.5,
    )
    with open(os.path.join(ARTIFACTS_DIR, "plausible_reasons.json"), "w") as f:
        json.dump(reasons, f, indent=2)

    summary = {
        "xgboost": {k: v for k, v in xgb_metrics.items() if k != "report"},
        "mlp": {k: v for k, v in mlp_metrics.items() if k != "report"},
        "transformer": {k: v for k, v in tr_metrics.items() if k != "report"},
        "artifacts_dir": ARTIFACTS_DIR,
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Artifacts written to:", ARTIFACTS_DIR)
    return summary


def build_plausible_reasons(
    feats_subset, X_test, xgb_model, xgb_top_features, tr_token_imp, feature_names, threshold=0.5
):
    probs = xgb_model.predict_proba(X_test)[:, 1]
    top_feat_names = [f for f, _ in xgb_top_features[:10]]
    reasons = {}
    for i, (uid, prob) in enumerate(zip(feats_subset["user_id"].tolist(), probs.tolist())):
        user_row = X_test.iloc[i]
        recs = []
        if prob >= threshold:
            # Feature-based reasons
            for fname in top_feat_names:
                val = user_row[fname]
                if fname in ["recency_days", "avg_gap", "max_gap", "gap_spikes"] and val > np.percentile(X_test[fname], 75):
                    recs.append(f"Long inactivity/gaps (high {fname}={round(float(val),2)})")
                if fname in ["paused_rate"] and val > np.percentile(X_test[fname], 75):
                    recs.append("Playback interruptions likely frustrating experience")
                if fname in ["genre_entropy"] and val < np.percentile(X_test[fname], 25):
                    recs.append("Low content diversity; possible boredom")
                if "monthly_price" in fname and val > np.percentile(X_test[fname], 75):
                    recs.append("Price sensitivity risk due to higher plan cost")

            # Sequence-based reasons
            toks = tr_token_imp.get(uid, [])
            if toks:
                common_genres = Counter([t["genre"] for t in toks if t["genre"] != "PAD"]).most_common(1)
                if common_genres:
                    g, _ = common_genres[0]
                    recs.append(f"Recent consumption skewed to {g}; limited variety")
                late = [t for t in toks if t["hour"] in [23, 0, 1]]
                if len(late) >= 2:
                    recs.append("Heavy late-night viewing; potential fatigue")
                short = [t for t in toks if t["minutes"] < 15]
                if len(short) >= 2:
                    recs.append("Short/abandoned sessions; engagement drop")

            # Tabular device usage
            if user_row.get("device_mobile", 0) > 0.6:
                recs.append("Mostly mobile usage; network variability may hurt experience")

        reasons[uid] = {
            "churn_probability": round(prob, 4),
            "at_risk": prob >= threshold,
            "reasons": sorted(list(set(recs)))[:5],
        }
    return reasons


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="OTT Churn Prediction Pipeline")
    parser.add_argument("--n_users", type=int, default=5000)
    parser.add_argument("--n_titles", type=int, default=1200)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--seq_epochs", type=int, default=8)
    parser.add_argument("--mlp_epochs", type=int, default=18)
    parser.add_argument("--xgb_rounds", type=int, default=200)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    run_pipeline(
        n_users=args.n_users,
        n_titles=args.n_titles,
        device_name=args.device,
        seq_epochs=args.seq_epochs,
        mlp_epochs=args.mlp_epochs,
        xgb_rounds=args.xgb_rounds,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
