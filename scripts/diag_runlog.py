"""Find the fine-grained early minimum of the adaptation angle_diff — where the decoder passes
through the correct compensation before over-rotating."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np

df = pd.read_csv("figures/paper/adaptation_run_log_seed_0_target_135.csv")
ad = df["angle_diff"].abs().values
w = 1000
print("ep : rolling|angle_diff| (first 30k)")
for i in range(0, min(30000, len(ad)), 2000):
    print("  %6d : %5.1f" % (i, float(np.mean(ad[max(0, i - w):i + 1]))))
allroll = np.array([np.mean(ad[max(0, i - w):i + 1]) for i in range(len(ad))])
print("min rolling angle_diff = %.1f at ep %d" % (allroll.min(), int(allroll.argmin())))
