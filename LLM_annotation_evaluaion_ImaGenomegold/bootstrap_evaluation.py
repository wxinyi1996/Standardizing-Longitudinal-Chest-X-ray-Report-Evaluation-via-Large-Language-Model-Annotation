import pandas as pd
import ast
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from bootstrap_modules import bootstrap
import pickle

def safe_first(x):
    if isinstance(x, str) and x.strip() != "":
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list) and len(val) > 0:
                return val[0]
            else:
                return None
        except Exception:
            return None
    return None

def cal(df):
    df = df.copy()

    df.loc[df["predict_label_name"].str.contains("device", na=False), "classification_prediction"] = 0
    df["status_predict_label_name"] = df["status_predict_label_name"].apply(safe_first)

    cond1 = (df["classification_prediction"] == 0) | (df["classification_prediction"] == '0')
    cond2 = df["status_predict_label_name"] == "unmentioned"

    df.loc[cond1 | cond2, "status_predict_label_name"] = 0

    df["gold_comparison"] = df["gold_comparison"].replace(["", np.nan], 0)

    classes = ['no change', 'improved', 'worsened']

    results = []

    total_TP = 0
    total_FP = 0
    total_FN = 0

    for cls in classes:
        y_true = df['gold_comparison'] == cls
        y_pred = df['status_predict_label_name'] == cls

        TP = ((y_true) & (y_pred)).sum()
        TN = ((~y_true) & (~y_pred)).sum()
        FP = ((~y_true) & (y_pred)).sum()
        FN = ((y_true) & (~y_pred)).sum()

        accuracy = (TP + TN) / len(df)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        total_TP += TP
        total_FP += FP
        total_FN += FN

        results.append({
            'class': cls,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    macro_f1 = np.mean([x['f1'] for x in results])
    f1_dict = {x['class']: x['f1'] for x in results}

    n_f1 = f1_dict['no change']
    i_f1 = f1_dict['improved']
    w_f1 = f1_dict['worsened']

    df['correct'] = df['status_predict_label_name'] == df['gold_comparison']

    micro_accuracy = df['correct'].mean()
    micro_precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    micro_recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    micro_f1 = (
        2 * micro_precision * micro_recall /
        (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0 else 0
    )

    return micro_accuracy, micro_f1, macro_f1, n_f1, i_f1, w_f1

def cal_silver(df):
    df = df.copy()

    df["gold_comparison"] = df["gold_comparison"].replace(["", np.nan], 0)
    df["silver_comparison"] = df["silver_comparison"].replace(["", np.nan], 0)

    classes = ['no change', 'improved', 'worsened']

    silver_results = []

    total_TP = 0
    total_FP = 0
    total_FN = 0

    for cls in classes:
        y_true = df['gold_comparison'] == cls
        y_pred = df['silver_comparison'] == cls

        TP = ((y_true) & (y_pred)).sum()
        TN = ((~y_true) & (~y_pred)).sum()
        FP = ((~y_true) & (y_pred)).sum()
        FN = ((y_true) & (~y_pred)).sum()

        total_TP += TP
        total_FP += FP
        total_FN += FN

        accuracy = (TP + TN) / len(df)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        silver_results.append({
            'class': cls,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    macro_f1 = np.mean([x['f1'] for x in silver_results])
    f1_dict = {x['class']: x['f1'] for x in silver_results}

    n_f1 = f1_dict['no change']
    i_f1 = f1_dict['improved']
    w_f1 = f1_dict['worsened']

    df['silver_correct'] = df['silver_comparison'] == df['gold_comparison']

    micro_accuracy = df['silver_correct'].mean()
    micro_precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    micro_recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    micro_f1 = (
        2 * micro_precision * micro_recall /
        (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0 else 0
    )

    return micro_accuracy, micro_f1, macro_f1, n_f1, i_f1, w_f1

def bootstrap_sample(gt):
    N = len(gt)
    
    indices = np.random.choice(
        N,
        size=N,
        replace=True
    )

    return indices


file_names = [
    './results/Qwen-32B-2.csv',
    './results/Qwen-72B-2.csv',
    './results/MedResearcher-R1-32B-2.csv',
    './results/Medgemma-27b-2.csv',
    './results/Llama-3.3-70B.csv'
]


predicts = []

for file_name in file_names:
    df = pd.read_csv(file_name)
    predicts.append(df)


# 任意一个csv中的silver都一样
silver_df = predicts[0]


model_names = [
    "Qwen-32B",
    "Qwen-72B",
    "MedResearcher-R1-32B",
    "Medgemma-27B",
    "Llama-3.3-70B",
    "Silver"
]


metric_names = [
    "micro_accuracy",
    "micro_f1",
    "macro_f1",
    "n_f1",
    "i_f1",
    "w_f1"
]


# =====================
# 原始结果
# =====================

original_scores = []

for df in predicts:
    scores = cal(df)
    original_scores.append(scores)


# silver原始结果
original_scores.append(
    cal_silver(silver_df)
)


original_scores = np.array(original_scores)


original_df = pd.DataFrame(
    original_scores,
    index=model_names,
    columns=metric_names
)
original_df.to_csv("./bootstrap_evaluation/original_df.csv")


# =====================
# Bootstrap
# =====================

num_boots = 1000

num_models = len(model_names)
num_metrics = len(metric_names)


boot_matrix = np.zeros(
    (
        num_boots,
        num_models,
        num_metrics
    )
)


for boot_idx in tqdm(range(num_boots)):


    # 所有模型共享sample
    indices = bootstrap_sample(
        predicts[0]
    )


    # ---------------------
    # 模型结果
    # ---------------------
    for model_idx, predict in enumerate(predicts):

        predict_boot = (
            predict.iloc[indices]
            .reset_index(drop=True)
        )


        scores = cal(
            predict_boot
        )


        boot_matrix[
            boot_idx,
            model_idx,
            :
        ] = scores



    # ---------------------
    # Silver
    # ---------------------

    silver_boot = (
        silver_df.iloc[indices]
        .reset_index(drop=True)
    )


    silver_scores = cal_silver(
        silver_boot
    )


    boot_matrix[
        boot_idx,
        -1,
        :
    ] = silver_scores



# 保存

with open(
    "./bootstrap_evaluation/boot_matrix.pkl",
    "wb"
) as f:
    pickle.dump(
        boot_matrix,
        f
    )


print("Bootstrap Finished.")


results = bootstrap(
    boot_matrix,
    original_scores,
    metric_names,
    model_names
)


with open(
    "./bootstrap_evaluation/boot_results.pkl",
    "wb"
) as f:
    pickle.dump(
        results,
        f
    )