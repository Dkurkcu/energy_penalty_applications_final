

import pandas as pd
import matplotlib.pyplot as plt

# === Load monitor.csv ===
df = pd.read_csv("C:/Users/90546/Desktop/reacher_v3/logs/monitor.csv", skiprows=1)

# === Plot episode reward over timesteps ===
plt.figure(figsize=(10, 5))
plt.plot(df["l"].cumsum(), df["r"], label="Episode Reward")
plt.xlabel("Timestep")
plt.ylabel("Episode Reward")
plt.title("PPO Training: Episode Reward over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
