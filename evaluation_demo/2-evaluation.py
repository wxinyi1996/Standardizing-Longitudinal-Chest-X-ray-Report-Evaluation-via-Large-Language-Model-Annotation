import json
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
import numpy as np
import re
import torch
from pprint import pprint
import ast
from sklearn.metrics import classification_report
from tqdm import tqdm
from itertools import combinations
import pickle


np.random.seed(42)

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def preprocess(gt_reports,predict_reports):
    gt_reports_list = [" ".join(clean_report_mimic_cxr(gt).split()[:100]) for gt in gt_reports]
    predict_reports_list = [" ".join(clean_report_mimic_cxr(pred).split()) for pred in predict_reports]

    gt_dict = {i: [gt] for i, gt in enumerate(gt_reports_list)}
    pred_dict = {i: [pred] for i, pred in enumerate(predict_reports_list)}

    return gt_dict,pred_dict


def process_status(predict, gt):

    predict = predict.copy()
    gt = gt.copy()

    choose_column_gt='status'
    choose_column_predict='predict_status'
    if choose_column_gt==None or choose_column_predict==None:
        raise NameError

    predict['predict_status'] = predict[choose_column_predict].apply(ast.literal_eval)
    predict['gt_status'] = gt[choose_column_gt].apply(ast.literal_eval)

    predict_status_list = []
    gt_status_list = []

    for row in predict.itertuples(index=False):

        visit_id = row.study_id

        # ===== 处理预测 =====
        for s in row.predict_status:

            if ('support devices' in s) or ('unmentioned' in s):
                continue

            parts = s.split('_')

            if len(parts) == 3:
                predict_status_list.append(
                    f"{visit_id}_{parts[1]}_{parts[2]}"
                )
            else:
                print("ERROR:", s)
                break

        for s in row.gt_status:

            if ('support devices' in s) or ('unmentioned' in s):
                continue

            parts = s.split('_')
            if len(parts) == 2:
                gt_status_list.append(
                    f"{visit_id}_{parts[0]}_{parts[1]}"
                )

            else:
                print("ERROR:", s)
                break

    return set(predict_status_list), set(gt_status_list)

def status_calcu(gt_status_set,predict_status_set):

    truepos = gt_status_set.intersection(predict_status_set)
    falseneg = gt_status_set.difference(predict_status_set)
    falsepos = predict_status_set.difference(gt_status_set)

    precision = len(truepos)/(len(truepos)+len(falsepos))
    recall = len(truepos)/(len(truepos)+len(falseneg))
    # print("len(truepos):",len(truepos))
    # print("len(falseneg):",len(falseneg))
    # print("len(falsepos):",len(falsepos))
    # print("micro-average disease progression:")
    # print('Precision:', precision)
    # print('Recall:', recall)
    # print('f1-score:', 2*precision*recall/(precision+recall))
    micro_f1 = 2*precision*recall/(precision+recall)

    categories = ["no change", "improved", "worsened"]

    metrics = {}

    for cat in categories:
        gt_cat = {x for x in gt_status_set if x.endswith(cat)}
        pred_cat = {x for x in predict_status_set if x.endswith(cat)}
        
        truepos = gt_cat.intersection(pred_cat)
        falseneg = gt_cat.difference(pred_cat)
        falsepos = pred_cat.difference(gt_cat)
        # print("categories:",cat)
        # print("len(truepos):",len(truepos))
        # print("len(falseneg):",len(falseneg))
        # print("len(falsepos):",len(falsepos))
        if len(truepos) + len(falsepos) == 0:
            precision = 0.0
        else:
            precision = len(truepos) / (len(truepos) + len(falsepos))
        
        if len(truepos) + len(falseneg) == 0:
            recall = 0.0
        else:
            recall = len(truepos) / (len(truepos) + len(falseneg))
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        metrics[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # # 打印结果
    # for cat, m in metrics.items():
    #     print(f"Category: {cat}")
    #     print(f"  Precision: {m['precision']:.4f}")
    #     print(f"  Recall:    {m['recall']:.4f}")
    #     print(f"  F1-score:  {m['f1']:.4f}")

    return micro_f1,metrics["no change"]['f1'], metrics["improved"]['f1'], metrics["worsened"]['f1']

def evaluate(predict, gt):
    assert gt['study_id'].tolist() == predict['study_id'].tolist()

    gt_L_reports = gt['longitudinal_description'].fillna(" ").tolist()
    pred_L_reports = predict['predicted_L_description'].fillna(" ").tolist()

    gt_dict, pred_dict = preprocess(gt_L_reports,pred_L_reports)
   
    test_met = compute_language_scores(gt_dict, pred_dict)
    # print('Longitudinal sentences:')
    # pprint(test_met)

    L_B4, L_ME = test_met['BLEU_4'], test_met['METEOR']

    predict_status_set, gt_status_set = process_status(predict, gt)
    
    AF1, NF1, IF1, WF1 = status_calcu(gt_status_set,predict_status_set)

    L_B4, L_ME, AF1, NF1, IF1, WF1 = [
        round(x * 100, 2) for x in 
        (L_B4, L_ME, AF1, NF1, IF1, WF1)
    ]

    return L_B4, L_ME, AF1, NF1, IF1, WF1



def compute_language_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Meteor(), "METEOR")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

def bootstrap_sample(gt):
    N = len(gt)
    
    indices = np.random.choice(
        N,
        size=N,
        replace=True
    )

    return indices

def calculate_original_scores(
    gt_path,
    predict_path
):

    gt_origin = pd.read_csv(gt_path)

    predict = pd.read_csv(predict_path)

    original_scores = []

    assert (gt_origin.study_id.tolist()== predict.study_id.tolist())

    scores = evaluate(
        predict,
        gt_origin
    )

    original_scores.append(scores)

    return original_scores

predict_path = './10_annotated.csv'

metric_names = [
    'L_B4',
    'L_ME',
    'AF1',
    'NF1',
    'IF1',
    'WF1'
]

num_boots = 1000

gt_path = './10_gt_Qwenlabeled.csv'

# ==========================================================
# Load Data
# ==========================================================

gt_origin = pd.read_csv(gt_path)

predicts = pd.read_csv(predict_path)

original_scores = calculate_original_scores(
    gt_path,
    predict_path
)

original_df = pd.DataFrame(
    original_scores,
    columns=metric_names
)

print('results:',original_df)

# results:    L_B4  L_ME    AF1   NF1  IF1  WF1
# 0  4.87  6.58  57.14  64.0  0.0  0.0