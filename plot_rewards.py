import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# === Load monitor.csv ===
df = pd.read_csv("C:/Users/90546/Desktop/reacher_v3/logs/v2.monitor.csv", skiprows=1)

# === Original: Plot episode reward over timesteps ===
plt.figure(figsize=(10, 5))
plt.plot(df["l"].cumsum(), df["r"], label="Episode Reward")
plt.xlabel("Timestep")
plt.ylabel("Episode Reward")
plt.title("PPO Training: Episode Reward over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# === New: Plot moving window success rate ===
SUCCESS_REWARD = 7.5  # What counts as "success" (adjust as needed)
WINDOW = 100          # Episodes per window

rewards = df["r"].values
success = (rewards >= SUCCESS_REWARD).astype(float)
success_rate = np.convolve(success, np.ones(WINDOW)/WINDOW, mode="valid")

plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(success_rate)) + WINDOW, success_rate)
plt.xlabel(f"Episode (window={WINDOW})")
plt.ylabel("Success Rate")
plt.title(f"Moving Success Rate (Reward ≥ {SUCCESS_REWARD})")
plt.ylim([0, 1.05])
plt.grid(True)
plt.tight_layout()
plt.show()

plt.hist(df['r'], bins=30)
plt.title("Distribution of Episode Rewards")
plt.xlabel("Reward")
plt.ylabel("Count")
plt.show()

