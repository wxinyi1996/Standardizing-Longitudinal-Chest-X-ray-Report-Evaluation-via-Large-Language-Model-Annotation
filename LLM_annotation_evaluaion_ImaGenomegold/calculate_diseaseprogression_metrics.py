import pandas as pd
import ast
import numpy as np

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
    df.loc[df["predict_label_name"].str.contains("device", na=False), "classification_prediction"] = 0
    df["status_predict_label_name"] = df["status_predict_label_name"].apply(safe_first)
    cond1 = (df["classification_prediction"] == 0) | (df["classification_prediction"] == '0')

    # 条件2：status_predict_label_name == "unmentioned"
    cond2 = df["status_predict_label_name"] == "unmentioned"

    # 统一替换为 0
    df.loc[cond1 | cond2, "status_predict_label_name"] = 0

    df["gold_comparison"] = df["gold_comparison"].replace(["", np.nan], 0)
    df["silver_comparison"] = df["silver_comparison"].replace(["", np.nan], 0)

   

    # 输出正确/错误的数量
    # print(df['correct'].value_counts())

    # 假设 df 已有 status_predict_label_name 和 gold_comparison
    # df = pd.DataFrame({...})

    classes = ['no change', 'improved', 'worsened']

    results = []
    silver_results = []

    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_silver_TP = 0
    total_silver_FP = 0
    total_silver_FN = 0

    for cls in classes:
        # 当前类别为正类，其余为负类
        y_true = df['gold_comparison'] == cls
        y_pred = df['status_predict_label_name'] == cls
        y_silver_pred = df['silver_comparison'] == cls
        
        # True Positive (TP): 预测为 cls 且真实为 cls
        TP = ((y_true) & (y_pred)).sum()
        # True Negative (TN): 预测不是 cls 且真实也不是 cls
        TN = ((~y_true) & (~y_pred)).sum()
        # False Positive (FP): 预测为 cls 但真实不是 cls
        FP = ((~y_true) & (y_pred)).sum()
        # False Negative (FN): 预测不是 cls 但真实为 cls
        FN = ((y_true) & (~y_pred)).sum()
        
        accuracy = (TP + TN) / len(df)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
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

        # True Positive (TP): 预测为 cls 且真实为 cls
        TP = ((y_true) & (y_silver_pred)).sum()
        # True Negative (TN): 预测不是 cls 且真实也不是 cls
        TN = ((~y_true) & (~y_silver_pred)).sum()
        # False Positive (FP): 预测为 cls 但真实不是 cls
        FP = ((~y_true) & (y_silver_pred)).sum()
        # False Negative (FN): 预测不是 cls 但真实为 cls
        FN = ((y_true) & (~y_silver_pred)).sum()

        total_silver_TP += TP
        total_silver_FP += FP
        total_silver_FN += FN

        accuracy = (TP + TN) / len(df)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        silver_results.append({
            'class': cls,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    metrics_df = pd.DataFrame(results)
    print("Per-class metrics (using pure Pandas):")
    print(metrics_df)

    metrics_df = pd.DataFrame(silver_results)
    print("silver results:")
    print(metrics_df)

    df['correct'] = df.apply(lambda x: x['status_predict_label_name'] == x['gold_comparison'], axis=1)
    df['silver_correct'] = df.apply(lambda x: x['silver_comparison'] == x['gold_comparison'], axis=1)

    micro_accuracy = df['correct'].mean()
    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print("Micro-level metrics:")
    print(f"Accuracy: {micro_accuracy:.4f}, Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1: {micro_f1:.4f}\n")

    micro_accuracy = df['silver_correct'].mean()
    micro_precision = total_silver_TP / (total_silver_TP + total_silver_FP) if (total_silver_TP + total_silver_FP) > 0 else 0
    micro_recall = total_silver_TP / (total_silver_TP + total_silver_FN) if (total_silver_TP + total_silver_FN) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print("silver Micro-level metrics:")
    print(f"Accuracy: {micro_accuracy:.4f}, Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1: {micro_f1:.4f}\n")



file_names = [
    './results/Qwen-32B-2.csv',
    './results/Qwen-72B-2.csv',
    './results/MedResearcher-R1-32B-2.csv',
    './results/Medgemma-27b-2.csv',
    './results/Llama-3.3-70B.csv',
]

for file_name in file_names:
    print(file_name)
    cal(pd.read_csv(file_name))
