import pandas as pd
import ast
import numpy as np


def cal_longitudinal_sentence_classification(df_a):
    print('Longitudinal Sentence Classification:')

    df_a.loc[df_a["predict_label_name"].str.contains("device", na=False), "classification_prediction"] = 0
    y_pred = ((df_a['classification_prediction'] == 1) | 
          (df_a['classification_prediction'] == '1')).astype(int)
    y_true = df_a['processed_gold_comparison'].astype(int)

    TP = ((y_pred == 1) & (y_true == 1)).sum()  # 真阳性
    TN = ((y_pred == 0) & (y_true == 0)).sum()  # 真阴性
    FP = ((y_pred == 1) & (y_true == 0)).sum()  # 假阳性
    FN = ((y_pred == 0) & (y_true == 1)).sum()  # 假阴性

    # 计算指标
    accuracy = (TP + TN) / len(df_a)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 输出结果 #72B #32B
    print(f"Accuracy: {accuracy:.3f}") 
    print(f"Precision: {precision:.3f}") 
    print(f"Recall: {recall:.3f}") 
    print(f"F1 Score: {f1:.3f}") 

    y_pred = (df_a['silver_comparison'].notna()).astype(int)
    TP = ((y_pred == 1) & (y_true == 1)).sum()  # 真阳性
    TN = ((y_pred == 0) & (y_true == 0)).sum()  # 真阴性
    FP = ((y_pred == 1) & (y_true == 0)).sum()  # 假阳性
    FN = ((y_pred == 0) & (y_true == 1)).sum()  # 假阴性

    # 计算指标
    accuracy = (TP + TN) / len(df_a)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print('silver:')
    print(f"Accuracy: {accuracy:.3f}") 
    print(f"Precision: {precision:.3f}") 
    print(f"Recall: {recall:.3f}") 
    print(f"F1 Score: {f1:.3f}") 



file_names = [
    './results/Qwen-32B-2.csv',
    './results/Qwen-72B-2.csv',
    './results/MedResearcher-R1-32B-2.csv',
    './results/Medgemma-27b-2.csv',
    './results/Llama-3.3-70B.csv',
]

for file_name in file_names:
    print(file_name)
    cal_longitudinal_sentence_classification(pd.read_csv(file_name))
