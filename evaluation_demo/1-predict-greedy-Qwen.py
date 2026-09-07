import json
import pandas as pd
import numpy
import re
import torch
from pprint import pprint
import ast
from randomLLM_annotation import Longitudinal_prediction, prediction_disease_status
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def evaluate(gt_path,predict_path,model,tokenizer):
    
    gt = pd.read_csv(gt_path)
    predict = pd.read_csv(predict_path)
    predict['predicted_Longitudinal_annotation'] = None
    predict['predicted_L_description'] = None
    predict['predict_status'] = None

    for i, row in tqdm(predict.iterrows(), total=10):
        predicted_L_description_list = []
        predict_status = []
        predict_report = str(row['report'])
        predicted_sentences, predicted_Longitudinal_annotation = Longitudinal_prediction(predict_report,model,tokenizer)
        disease_list = ast.literal_eval(gt.loc[i, 'status'])
        for j, (ann, sentence) in enumerate(zip(predicted_Longitudinal_annotation, predicted_sentences)):
            if str(ann) == "1":
                predicted_L_description_list.append(sentence.strip())
                
                for status in disease_list:
                    disease = status.split('_')[0]
                    result, response = prediction_disease_status(tokenizer=tokenizer,model=model,sentence=sentence,disease=disease)
                    if len(result)>0:
                        predict_status.append(str(j)+'_'+disease+'_'+result[0])
                    else:
                        predict_status.append(str(j)+'_'+disease+'_unkown')


        predicted_L_description = ". ".join(predicted_L_description_list)

        predict.at[i, 'predicted_Longitudinal_annotation'] = predicted_Longitudinal_annotation
        predict.at[i, 'predicted_L_description'] = predicted_L_description
        predict.at[i, 'predict_status'] = predict_status

        save_path = '10_annotated.csv'
        print('save_path:', save_path)
        # 保存
        predict.to_csv(save_path, index=False)


my_token = 'xxxx'
# Note: The default behavior now has injection attack prevention off.
model_name = "xx/Qwen2.5-32B-Instruct"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2", token=my_token).eval()
gt_path = './10_gt_Qwenlabeled.csv'

predict = './10_results.csv'
print("model:",predict)
evaluate(gt_path,predict,model,tokenizer)