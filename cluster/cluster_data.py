
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import lstm_utils

# Klaszterek létrehozása
def load_and_cluster(csv_file="df_test_10k.csv", n_clusters=4):
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df_processed = lstm_utils.preprocess_dataframe(df)

    user_features = df_processed.groupby("uid").agg(
        mean_rg=("radius_of_gyration", "mean"),
        std_rg=("radius_of_gyration", "std"),
        mean_daily_rg=("daily_radius_of_gyration", "mean"),
        uniq_cells=("unique_cells_count", "mean"),
        frac_home=("is_home", "mean"),
        frac_work=("is_workplace", "mean"),
        weekend_ratio=("is_weekend", "mean"),
        # mean_hour_sin=("hour_sin", "mean"),
        # mean_hour_cos=("hour_cos", "mean"),
        # mean_temp=("temperature", "mean"),
        # std_temp=("temperature", "std"),
        # mean_rain=("rain", "mean")
    ).fillna(0)

    #skálázás
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_features)

    # 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_features["cluster"] = kmeans.fit_predict(X_scaled)

    df_proc_wclu = df_processed.merge(user_features[["cluster"]], on="uid", how="left")
    return df_proc_wclu
