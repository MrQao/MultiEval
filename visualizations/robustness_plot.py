import matplotlib.pyplot as plt

def plot_robustness(summary, output_path):
    radii = [r["radius"] for r in summary]
    accuracy = [r["accuracy"] for r in summary]
    invalid_rate = [r["invalid_rate"] for r in summary]
    flip_rate = [r["flip_rate"] for r in summary]

    plt.figure(figsize=(8, 5))

    plt.plot(radii, accuracy, marker="o", label="Accuracy")
    plt.plot(radii, invalid_rate, marker="s", label="Invalid Rate")

    # Flip Rate 可能为 None，保留缺失值而不是当作 0
    plt.plot(radii, flip_rate, marker="^", label="Flip Rate")

    plt.xlabel("Gaussian Blur Radius")
    plt.ylabel("Rate")
    plt.title("Qwen3-VL Robustness on POPE")

    plt.xticks(radii)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()