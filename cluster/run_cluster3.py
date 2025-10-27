import pandas as pd
import lstm_utils
from cluster_data import load_and_cluster


CLUSTER_ID = 3

# Adatok betöltése + klaszterezés
df_proc_wclu = load_and_cluster("df_test_10k.csv", n_clusters=4)

# Kiválasztott klaszter
df_clu = df_proc_wclu[df_proc_wclu["cluster"] == CLUSTER_ID]

print(f"\n=== Klaszter {CLUSTER_ID} ===")
res = lstm_utils.run_cluster_model_lstm_multi(df_clu, seq_len=20, epochs=20)

if res:
    res["cluster"] = CLUSTER_ID
    out_file = f"results_cluster{CLUSTER_ID}.csv"
    pd.DataFrame([res]).to_csv(out_file, index=False)
    print(f"Eredmény elmentve: {out_file}")
