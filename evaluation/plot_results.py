import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_plots():
    summary = pd.read_csv("results/summary_comparison.csv")
    logs = pd.read_csv("results/full_benchmark_logs.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#7f8c8d", "#3498db", "#2ecc71", "#e67e22"]

    # --- ГРАФИК 1: Качество и Задержка (P50 / P95) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    archs = summary["Архитектура"]
    scores = summary["Среднее качество (1-5)"]

    bars = ax1.bar(archs, scores, color=colors, width=0.55)
    ax1.set_title("Среднее качество ответов (шкала 1–5)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 5.5)
    ax1.set_ylabel("Балл (LLM-as-a-Judge)")
    ax1.tick_params(axis="x", rotation=15)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}", ha="center", fontweight="bold")

    x = np.arange(len(archs))
    width = 0.35
    b1 = ax2.bar(x - width / 2, summary["Latency P50 (сек)"], width, label="P50 (Медиана)", color="#3498db")
    b2 = ax2.bar(x + width / 2, summary["Latency P95 (сек)"], width, label="P95 (95-й перцентиль)", color="#e74c3c")
    ax2.set_title("Задержка ответа: P50 vs P95 (сек)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(archs, rotation=15)
    ax2.set_ylabel("Секунды (меньше — лучше)")
    ax2.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f}s", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("results/1_quality_and_latency.png", dpi=200)
    plt.close()

    # --- ГРАФИК 2: Pareto-фронт (Стоимость vs Качество) ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    costs_per_1k = summary["Средняя стоимость ($)"] * 1000  # переводим в $ за 1000 обращений для наглядности

    for i, row in summary.iterrows():
        c_1k = row["Средняя стоимость ($)"] * 1000
        q = row["Среднее качество (1-5)"]
        size = row["Latency P95 (сек)"] * 120
        ax.scatter(c_1k, q, s=size, color=colors[i], alpha=0.85, edgecolors="black", linewidth=1.5, zorder=3)
        ax.annotate(
            f"{row['Архитектура']}\n(${c_1k:.2f}/1k запр., P95={row['Latency P95 (сек)']}s)",
            (c_1k, q),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title("Компромисс: Стоимость 1000 обращений vs Качество\n(размер круга пропорционален задержке P95)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Стоимость 1000 обращений ($ USD) — левее дешевле")
    ax.set_ylabel("Среднее качество ответов (1–5) — выше лучше")
    ax.set_ylim(3.0, 5.2)
    ax.set_xlim(0, max(costs_per_1k) * 1.25)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("results/2_pareto_frontier.png", dpi=200)
    plt.close()

    # --- ГРАФИК 3: Качество в разрезе категорий обращений ---
    pivot = logs.pivot_table(index="category", columns="architecture", values="quality_score", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(11, 5), color=colors, width=0.75)
    ax.set_title("Качество ответов по категориям обращений", fontsize=12, fontweight="bold")
    ax.set_ylabel("Средний балл (1–5)")
    ax.set_xlabel("Категория обращения")
    ax.set_ylim(0, 5.6)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Архитектура", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig("results/3_quality_by_category.png", dpi=200)
    plt.close()

    print("Графики успешно сохранены в папку results/:")
    print(" - results/1_quality_and_latency.png")
    print(" - results/2_pareto_frontier.png")
    print(" - results/3_quality_by_category.png")

if __name__ == "__main__":
    generate_plots()