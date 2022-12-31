import json

import matplotlib.pyplot as plt
import numpy as np

with open("results_ffnn_.json", "r") as f:
    results = json.load(f)

for i, result in enumerate(results):
    acc_scores, auc_scores, train_times, loss_histories = result
    avg_acc = np.mean(acc_scores)
    avg_auc = np.mean(auc_scores)
    avg_train_time = np.mean(train_times)
    fig, ax = plt.subplots()
    for loss_hist in loss_histories:
        ax.plot(list(range(len(loss_hist))), loss_hist)
    ax.set_title(f"{i+1}, {avg_acc:.2%} Acc, {avg_auc:.2%} Auc, {avg_train_time:.2f}s to train")
    plt.show()
