# ===============================
# MASTER PIPELINE – KLASZTER LSTM
# ===============================

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

# Mixed precision és GPU memória kezelés
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# --- Segédfüggvények ---

def topk_acc_numpy(y_true, y_proba, k=5):
    if y_proba.ndim != 2:
        raise ValueError("y_proba 2D mátrix legyen (n_samples, n_classes).")

    n_classes = y_proba.shape[1]
    if n_classes == 0:
        return 0.0
    
    k_eff = min(k, n_classes)
    topk_idx = np.argpartition(y_proba, -k_eff, axis=1)[:, -k_eff:]
    correct = (topk_idx == y_true[:, None]).any(axis=1)
    return float(correct.mean())

def train_test_split_seq(X_list, y, split_ratio=0.8):
    n = len(y) 
    split = int(n * split_ratio)  
    
    X_train = [X[:split] for X in X_list]
    X_test  = [X[split:] for X in X_list]

    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test

def preprocess_dataframe(df, coarsen_factor=2):
    df = df.copy()

    df["hour"] = df["t"].astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 48)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 48)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

    num_cols = ["temperature", "rain", "daily_radius_of_gyration",
                "radius_of_gyration", "unique_cells_count"]

    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0

    df[num_cols] = df[num_cols].astype(float)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
# --- Szekvenciák felépítése -> 10 hogy gyorsítsam ---

def build_sequences_for_user(user_df, seq_len=20):
    df = user_df.copy().reset_index(drop=True)


    df["target"] = df["cell_id"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True) 

    y_all, uniques = pd.factorize(df["target"].astype(int))
    df["target_enc"] = y_all.astype(np.int32)
    num_classes = len(uniques) 

    cell_all, cell_uniques = pd.factorize(df["cell_id"].astype(int))
    df["cell_enc"] = cell_all.astype(np.int32)
    cell_vocab_size = int(df["cell_enc"].max()) + 2 

    df["dayofweek_enc"], _ = pd.factorize(df["days_of_week"])

    hour_ids = df["hour"].astype(np.int32).values
    num_cols = [
        "temperature", "rain", "daily_radius_of_gyration",
        "radius_of_gyration", "unique_cells_count", "fraction_missing",
        "hour_sin", "hour_cos", "is_weekend",
        "is_home", "is_workplace",
        "dayofweek_enc"
    ]
    num_feats = df[num_cols].astype(np.float32).values

    X_cell, X_hour, X_num, y = [], [], [], []
    for i in range(len(df) - seq_len):
        sl = slice(i, i + seq_len)
        X_cell.append(df["cell_enc"].values[sl])
        X_hour.append(hour_ids[sl])
        X_num.append(num_feats[sl, :])
        y.append(df["target_enc"].values[i + seq_len - 1])

    if len(y) == 0:
        return None

    return (np.asarray(X_cell, dtype=np.int32),
            np.asarray(X_hour, dtype=np.int32),
            np.asarray(X_num,  dtype=np.float32),
            np.asarray(y,      dtype=np.int32),
            cell_vocab_size, 48, num_classes)  

# --- Modellépítés ---

def build_lstm_multi(num_cells, num_hours, num_numeric, num_classes):
    inp_cell = layers.Input(shape=(None,), dtype="int32", name="cell_in")

    emb_cell = layers.Embedding(num_cells, 64, mask_zero=False)(inp_cell)
    x_cell = layers.LSTM(128, recurrent_dropout=0.1)(emb_cell)  

    inp_hour = layers.Input(shape=(None,), dtype="int32", name="hour_in")
    emb_hour = layers.Embedding(num_hours, 16, mask_zero=False)(inp_hour)
    x_hour = layers.LSTM(32)(emb_hour)  

    inp_num = layers.Input(shape=(None, num_numeric), dtype="float32", name="num_in")
    x_num = layers.LSTM(32)(inp_num)

    x = layers.Concatenate()([x_cell, x_hour, x_num])

    x = layers.Dense(128, activation="relu")(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=[inp_cell, inp_hour, inp_num], outputs=out)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# --- Klaszter szekvenciák építése ---
def build_sequences_for_cluster(cluster_df, seq_len=20):
    all_X_cell, all_X_hour, all_X_num, all_y = [], [], [], []
    cell_vocab_size, num_classes = 0, 0

    # a klaszter minden felhasználójára szekvenciákat épít
    for uid, dfu in cluster_df.groupby("uid"):
        seqs = build_sequences_for_user(dfu, seq_len=seq_len)
        if seqs is None: continue
        X_cell, X_hour, X_num, y, cell_vocab, hour_vocab, n_classes = seqs
        all_X_cell.append(X_cell)
        all_X_hour.append(X_hour)
        all_X_num.append(X_num)
        all_y.append(y)
        cell_vocab_size = max(cell_vocab_size, cell_vocab)
        num_classes = max(num_classes, n_classes)

    # Ha nincs adat a klaszterben, None-t ad vissza, ha van, akkor összefűzi az összes felhasználó szekvenciáját
    if len(all_y) == 0:
        return None
    return (np.vstack(all_X_cell),
            np.vstack(all_X_hour),
            np.vstack(all_X_num),
            np.concatenate(all_y),
            cell_vocab_size, hour_vocab, num_classes)

# --- Klaszter modell futtatása ---
def run_cluster_model_lstm_multi(df_cluster, seq_len=20, epochs=20, cluster_id=None):
    # adott klaszter szekvenciáinak felépítése
    seqs = build_sequences_for_cluster(df_cluster, seq_len=seq_len)
    if seqs is None: return None
    X_cell, X_hour, X_num, y, cell_vocab, hour_vocab, num_classes = seqs

    # adatok felosztása tanító és teszt részre
    X_train, X_test, y_train, y_test = train_test_split_seq([X_cell, X_hour, X_num], y, split_ratio=0.8)

    # modell felépítése 
    model = build_lstm_multi(cell_vocab, hour_vocab, X_num.shape[2], num_classes)
    # korai leállítás beállítása
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # modell tanítása
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=512,
        verbose=1,
        callbacks=[early_stop]
    )

    # --- Tanulási görbék megjelenítése ---
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Klaszter {cluster_id} - Loss görbe")
    plt.show()

    # --- Kiértékelés a teszt halmazon ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Klaszter {cluster_id}] Test accuracy: {test_acc:.3f}, Test loss: {test_loss:.3f}")

    # --- Predikciók és top-k pontosság ---
    proba_test = model.predict(X_test, verbose=0)
    pred_test = np.argmax(proba_test, axis=1)
    acc = accuracy_score(y_test, pred_test)
    top5 = topk_acc_numpy(y_test, proba_test, k=5)

    # --- Eredmények összefoglalása ---
    return {
        "cluster_id": cluster_id,
        "model_type": "LSTM-multi",
        "num_classes": num_classes,
        "train_accuracy": history.history.get("accuracy", [None])[-1],
        "val_accuracy": history.history.get("val_accuracy", [None])[-1],
        "test_accuracy": test_acc,
        "accuracy": acc,
        "top5_accuracy": top5,
        "epochs_ran": len(history.history["loss"]),
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "best_val_loss": min(history.history["val_loss"]),
        "best_epoch": np.argmin(history.history["val_loss"]) + 1,
        "num_samples": len(df_cluster)
    }




