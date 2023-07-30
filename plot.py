import pandas as pd
import matplotlib.pyplot as plt
sample_df = pd.read_csv ('train_log.csv')
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sample_df["mean"] = sample_df["collisions"].rolling(10).mean()
sample_df["std"] = sample_df["collisions"].rolling(10).std()
ax.fill_between(sample_df.index,
                sample_df["mean"]-sample_df["std"],
                sample_df["mean"]+sample_df["std"],
                alpha=0.2)
ax.plot(sample_df.index,sample_df["mean"], '-', label="collisions")

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("training_collisions.png")