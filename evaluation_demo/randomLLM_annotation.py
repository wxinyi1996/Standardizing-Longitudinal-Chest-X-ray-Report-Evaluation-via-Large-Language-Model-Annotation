from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import pandas as pd
import json
import os, re
import ast
import numpy as np
from tqdm import tqdm
import copy
import torch

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

def process_status_prompt(sentence,disease):
    prompt_1 = 'The following sentence \''+sentence +'\''
    prompt_1_1 =  'describes the status of \''+ disease +'\' as <no change> (e.g., similar, unchanged), <improved> (e.g., resolved), or <worsened>. Return the answer in the form of <no change>, <improved>, or <worsened>. If this condition is not mentioned, return <unmentioned>.'
    prompt_2 = 'Note that if \'change\' or \'new\' is mentioned along with a newly appearing lesion, it usually implies worsening. Example: in cases of \'pleural effusion\', \'New small bilateral pleural effusions.\' means <worsened>. In cases of \'low lung volumes\', \'lower\' means <worsened>; \'increased\' means <improved>. '
    prompt = prompt_1+prompt_1_1+prompt_2
    return prompt


def process_disease_prompt(sentence):
    prompt_1 = 'Which item in the list most related to the following sentence \''+sentence +'\''
    prompt_1_1 = '''? The list is ['support devices','airspace opacity', 'alveolar hemorrhage', 'aspiration', 'atelectasis', 'bone', 'bronchiectasis', 'calcified nodule', 'clavicle fracture', 'consolidation', 'copd/emphysema', 'costophrenic angle blunting', 'elevated hemidiaphragm', 'enlarged cardiac silhouette', 'enlarged hilum', 'fluid overload/heart failure', 'goiter', 'granulomatous disease', 'hernia', 'hydropneumothorax', 'increased reticular markings/ild pattern', 'infiltration', 'interstitial lung disease', 'lobar/segmental collapse', 'low lung volumes', 'lung cancer', 'lung lesion', 'lung opacity', 'mass/nodule (not otherwise specified)', 'mediastinal displacement', 'mediastinal widening', 'opacity', 'pericardial effusion', 'pleural effusion', 'pleural/parenchymal scarring', 'pneumonia', 'pneumothorax', 'pulmonary edema/hazy opacity', 'rib fracture', 'scoliosis', 'shoulder osteoarthritis', 'spinal degenerative changes', 'spinal fracture', 'sub-diaphragmatic air', 'subcutaneous air', 'superior mediastinal mass/enlargement', 'tortuous aorta', 'vascular calcification', 'vascular congestion', 'vascular redistribution'] Please reply with the answer enclosed in <>, for example <spinal fracture>. Return <support devices> for any changes in support device positions.'''
    prompt = prompt_1+prompt_1_1
    return prompt

def process_longitudinal_prompt(sentence):
    prompt_1 = 'Given a radiology report sentence \''+sentence +'\''
    prompt_1_1 = ''', determine if it compares the current image with prior studies (e.g., remain, compare, similar, stable, increased, still, new, again). If yes, return <1>. If no, return <0>. Examples: "Cardiac and mediastinal silhouettes are stable." returns <1>, ''No larger pleural effusions.'' returns <1>,''Increased retrocardiac opacity may reflect atelectasis.'' returns <1>.'''
    prompt = prompt_1+prompt_1_1
    return prompt

def prediction_disease_status(tokenizer,model,sentence,disease):
    prompt = process_status_prompt(sentence,disease)
    response = prediction(tokenizer,model,prompt)
    result = re.findall(r'<(.*?)>', response)
    if result == []:
        result=response
    return result,response

def prediction_disease(tokenizer,model,sentence):
    prompt = process_disease_prompt(sentence)
    response = prediction(tokenizer,model,prompt)
    result = re.findall(r'<(.*?)>', response)
    return result,response

def prediction_longitudinal(tokenizer,model,sentence):
    prompt = process_longitudinal_prompt(sentence)
    response = prediction(tokenizer,model,prompt)
    if '<1>' in response:
        result='1'
    elif '<0>' in response:
        result='0'
    else:
        result= copy.deepcopy(response)
    if ('<1>' in response) and ('<0>' in response):
        result= copy.deepcopy(response)
    return result, response


def prediction(tokenizer,model,prompt):
    # prepare the model input
    messages = [{'role':'system','content':'You are a medical AI assistant.'},{'role':'user','content':prompt}]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt",padding=True).to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,      # 开启采样
        temperature=0.7,     # 控制随机程度
        top_p=0.9,           # 核采样，过滤长尾低概率词
        top_k=20             # 进一步限制前50个最高概率词
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def post_process(tokenizer,model,answers):
    prompt = 'I got an answer from another large language model. The answer is \''+ answers +'\'. It is too long. I just want to know if the final answer is <0> or <1>. Can you tell me?'
    response = prediction(tokenizer,model,prompt)
    if '<1>' in response:
        result='1'
    elif '<0>' in response:
        result='0'
    else:
        print('2stage, error,no <0> or <1>')
        print('prompt:',prompt)
        print('response:',response)
        result= copy.deepcopy(response)
    if ('<1>' in response) and ('<0>' in response):
        print('2stage, error, both <0> and <1> in the sentence')
        print('prompt:',prompt)
        print('response:',response)
        result= copy.deepcopy(response)   
    return result

def Longitudinal_prediction(predicted_report,model,tokenizer):
    predicted_report = clean_report_mimic_cxr(predicted_report)
    predicted_sentences = [s.strip() for s in predicted_report.split('.') if s.strip()]
    predicted_Longitudinal_annotation = []
    for sentence in predicted_sentences:
        result, response = prediction_longitudinal(tokenizer=tokenizer,model=model,sentence=sentence)
        if result!='0' and result!='1':
            result = post_process(tokenizer=tokenizer,model=model,answers=result)
        predicted_Longitudinal_annotation.append(result)
    return predicted_sentences, predicted_Longitudinal_annotation

if __name__ == '__main__':
    my_token = 'xxxx'
    # Note: The default behavior now has injection attack prevention off.

    model_name = "xxx/Qwen2.5-32B-Instruct"
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2", token=my_token).eval()

    print('model.config._attn_implementation:', model.config._attn_implementation)
    all_samples = pd.read_csv('./bbox_disease_comparison_allsentencegoldsilver.csv')
    all_samples['sentence'] = all_samples['silver_sentence'].fillna(all_samples['gold_sentence'])
    all_samples['classification_prediction'] = None
    all_samples['processed_gold_comparison'] = None
    all_samples['predict_label_name'] = None
    all_samples['status_predict_label_name'] = None
    for i,row in tqdm(all_samples.iterrows(),total=1975):
        sentence = row['sentence']

        result, response = prediction_longitudinal(tokenizer=tokenizer,model=model,sentence=sentence)
        if result!='0' and result!='1':
            result = post_process(tokenizer=tokenizer,model=model,answers=result)
        if type(row['gold_comparison'])!=str:
            processed_gold_comparison = '0'
        else:
            processed_gold_comparison = '1' 
        all_samples.at[i, 'classification_prediction'] = result
        all_samples.at[i, 'processed_gold_comparison'] = processed_gold_comparison
        if result=='1':
            result, response = prediction_disease(tokenizer=tokenizer,model=model,sentence=sentence)
            all_samples.at[i, 'predict_label_name'] = result
            disease = result[0]
            result, response = prediction_disease_status(tokenizer=tokenizer,model=model,sentence=sentence,disease=disease)
            all_samples.at[i, 'status_predict_label_name'] = result
    all_samples.to_csv('./results/no_sample/Qwen2.5-32B.csv', index=False)
