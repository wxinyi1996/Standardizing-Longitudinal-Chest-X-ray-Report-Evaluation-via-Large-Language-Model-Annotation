import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 全局绘图样式（Nature 推荐使用 Arial 或 Helvetica 字体）
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def build_pairwise_matrix(
    data, models, val_col="MeanDiff", is_prob=False, fill_diag=0.0
):
    mat = pd.DataFrame(index=models, columns=models, dtype=float)
    np.fill_diagonal(mat.values, fill_diag)

    for _, row in data.iterrows():
        m1, m2, val = row["Model1"], row["Model2"], row[val_col]
        mat.loc[m1, m2] = val
        if not is_prob:
            mat.loc[m2, m1] = -val
        else:
            mat.loc[m2, m1] = 1.0 - val
    return mat


# 读取数据
df = pd.read_csv("./paired_bootstrap_test.csv")

# 模型名称简写映射
name_mapping = {
    "Longitudinal-R2Gen": "L-R2Gen",
}

df["Model1"] = df["Model1"].replace(name_mapping)
df["Model2"] = df["Model2"].replace(name_mapping)

metrics = ["AF1",'NF1','IF1','WF1']

for metric_name in metrics:
    sub_df = df[df["Metric"] == metric_name]

    # 获取排序后的模型列表
    models = sorted(list(set(sub_df["Model1"]).union(set(sub_df["Model2"]))))

    # 生成数据矩阵
    diff_mat = build_pairwise_matrix(
        sub_df, models, val_col="MeanDiff", is_prob=False, fill_diag=0.0
    )
    prob_mat = build_pairwise_matrix(
        sub_df,
        models,
        val_col="Prob(Model1>Model2)",
        is_prob=True,
        fill_diag=0.5,
    )

    # 仅保留下三角掩码（k=1 保留对角线）
    mask_diff = np.triu(np.ones_like(diff_mat, dtype=bool), k=1)
    mask_prob = np.triu(np.ones_like(prob_mat, dtype=bool), k=1)

    # -------------------------------------------------------------
    # 1. 独立绘制并保存 Mean Difference 热力图
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    if metric_name=="AF1":
        sns.heatmap(
            diff_mat,
            mask=mask_diff,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 9},
            cmap="vlag",
            vmin=-23,
            vmax=23,
            center=0,
            cbar_kws={
                "label": f"Mean Difference (ModelA - ModelB)",
                "shrink": 0.8,
            },
            square=True,
        )
    else:
        hm = sns.heatmap(
            diff_mat,
            mask=mask_diff,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 9},
            cmap="vlag",
            vmin=-23,
            vmax=23,
            center=0,
            cbar_kws={
                "label": f" ",
                "shrink": 0.8,
            },
            square=True,
        )

    if metric_name=="AF1":
        ax.set_ylabel("Model A", fontsize=10, fontweight="bold")
        ax.set_xlabel("Model B", fontsize=10, fontweight="bold")
    else:
        ax.set_ylabel(" ", fontsize=10, fontweight="bold")
        ax.set_xlabel(" ", fontsize=10, fontweight="bold")

    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(
        f"{metric_name}_mean_diff_heatmap.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{metric_name}_mean_diff_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # -------------------------------------------------------------
    # 2. 独立绘制并保存 Win Probability 热力图
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    if metric_name=="AF1":
        sns.heatmap(
            prob_mat,
            mask=mask_prob,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 9},
            cmap="RdBu_r",
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={
                "label": "Win Probability P (ModelA > ModelB)",
                "shrink": 0.8,
            },
            square=True,
        )
    else:
        hm = sns.heatmap(
            prob_mat,
            mask=mask_prob,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 9},
            cmap="RdBu_r",
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={
                "label": f" ",
                "shrink": 0.8,
            },
            square=True,)

    if metric_name=="AF1":
        ax.set_ylabel("Model A", fontsize=10, fontweight="bold")
        ax.set_xlabel("Model B", fontsize=10, fontweight="bold")
    else:
        ax.set_ylabel(" ", fontsize=10, fontweight="bold")
        ax.set_xlabel(" ", fontsize=10, fontweight="bold")        
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(
        f"{metric_name}_win_prob_heatmap.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{metric_name}_win_prob_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()