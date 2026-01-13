import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===============================
# Configuration
# ===============================
LOG_FILE = "code/chatbot_evaluation_log.csv"
OUTPUT_DIR = "results"

plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid", context="talk")

# ===============================
# Load + Aggregate
# ===============================
def load_data():
    df = pd.read_csv(LOG_FILE)

    df = df.groupby(['Agent', 'Episode']).agg({
        'Step_Reward': 'sum',
        'Step_Cost': 'sum',
        'Final_WRS': 'max',
        'Final_DIE': 'max',
        'Final_Comp_Reward': 'max',
        'Final_Quality': 'max'
    }).reset_index()

    df.rename(columns={
        'Step_Reward': 'Total Reward',
        'Step_Cost': 'Workflow Cost',
        'Final_WRS': 'WRS',
        'Final_DIE': 'DIE',
        'Final_Comp_Reward': 'Compression Reward',
        'Final_Quality': 'Correctness'
    }, inplace=True)

    return df

# ===============================
# 1. Violin Distribution Plots
# ===============================
def plot_violin(df, metric):
    plt.figure(figsize=(10,6))
    sns.violinplot(
        data=df,
        x="Agent",
        y=metric,
        inner="quartile",
        cut=0,
        bw_adjust=1.5
    )
    min_r = df["Total Reward"].min()
    max_r = df["Total Reward"].max()

    plt.ylim(min_r, max_r)

    # plt.ylim(-1.0, 1.0)
    plt.title(f"Distribution of {metric} Across Episodes")
    plt.ylabel(f"{metric} (Higher is Better)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/violin_{metric.replace(' ','_')}.png", dpi=300)
    plt.close()

# ===============================
# 2. Pareto Scatter
# ===============================
def plot_pareto(df):
    means = df.groupby("Agent").mean(numeric_only=True).reset_index()

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=means,
        x="Workflow Cost",
        y="Total Reward",
        hue="Agent",
        s=180
    )

    # Ideal reference point
    ideal_reward = df["Total Reward"].max()
    ideal_cost = df["Workflow Cost"].min()

    plt.scatter(
        ideal_cost,
        ideal_reward,
        marker="*",
        s=300,
        c="black",
        label="Ideal (High Reward, Low Cost)",
        zorder=5
    )

    plt.title("Pareto Trade-off: Reward vs Cost")
    plt.xlabel("Workflow Cost (Lower is Better)")
    plt.ylabel("Total Reward (Higher is Better)")

    plt.legend(
        fontsize=14,
        title_fontsize=15,
        markerscale=0.8,
        loc="upper right"
    )

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/pareto_reward_cost.png", dpi=300)
    plt.close()

# ===============================
# 3. Episode-Bin Heatmaps
# ===============================
def plot_heatmap(df, metric, bins=10):
    df = df.copy()
    df["Episode Bin"] = pd.cut(df["Episode"], bins=bins)

    binned = (
        df.groupby(["Agent", "Episode Bin"])[metric]
          .mean()
          .reset_index()
    )

    pivot = binned.pivot(
        index="Agent",
        columns="Episode Bin",
        values=metric
    )

    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, cmap="viridis")
    plt.title(f"{metric} Across Episode Phases")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_{metric.replace(' ','_')}.png", dpi=300)
    plt.close()


# ===============================
# 4. Stability (Rolling Variance)
# ===============================
def plot_stability(df, metric, window=25):
    df_sorted = df.sort_values("Episode")
    df_sorted["Rolling STD"] = (
        df_sorted.groupby("Agent")[metric]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df_sorted,
        x="Episode",
        y="Rolling STD",
        hue="Agent"
    )
    plt.title(f"Stability of {metric} (Rolling Variability)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stability_{metric.replace(' ','_')}.png", dpi=300)
    plt.close()

# ===============================
# 5. Radar Chart (Summary)
# ===============================
def plot_radar(df):
    metrics = [
        'Total Reward',
        'Workflow Cost',
        'WRS',
        'DIE',
        'Correctness'
    ]

    means = df.groupby("Agent")[metrics].mean()

    # Normalize (invert where lower is better)
    norm = (means - means.min()) / (means.max() - means.min())
    norm['Workflow Cost'] = 1 - norm['Workflow Cost']
    norm['WRS'] = 1 - norm['WRS']
    norm['DIE'] = 1 - norm['DIE']

    labels = metrics
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)

    for agent in norm.index:
        values = norm.loc[agent].tolist()
        values += values[:1]
        ax.plot(angles, values, label=agent)
        ax.fill(angles, values, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Overall Agent Profile")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/radar_agents.png", dpi=300)
    plt.close()

# ===============================
# Main
# ===============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    # Violin plots (main figures)
    for m in ["Total Reward", "WRS", "DIE", "Workflow Cost", "Correctness"]:
        plot_violin(df, m)

    # Pareto
    plot_pareto(df)

    # Heatmaps
    for m in ["Total Reward", "WRS", "DIE"]:
        plot_heatmap(df, m)

    # Stability
    plot_stability(df, "Total Reward")

    # Radar
    plot_radar(df)

if __name__ == "__main__":
    main()
