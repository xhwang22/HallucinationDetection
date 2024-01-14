#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import io
import argparse
import torch
import scipy.stats
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from tqdm import tqdm
from utils import (
    get_data, 
    get_entailment_score,
    split_text,
)
from NBC_feature import get_NBC_features
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Evaluate hallucination detection task on retrieved documents"
    )
    
    parser.add_argument(
        "--model_name",
        type = str,
        default = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help = "The model path used to receive promises and hypotheses to output entailment scores.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type = str,
        default = None,
        help = "Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--test_file_path",
        type = str,
        default = "dataset/selfcheckgpt/dataset.json",
        help = "The file path of the selfcheckgpt, used to evaluate.",
    )
    parser.add_argument(
        "--webpage_file_path",
        type = str,
        default = "dataset/webpage",
        help = "The file path of the retrieved webpages.",
    )
    parser.add_argument(
        "--decomposed_file_path",
        type = str,
        default = "dataset/decomposed",
        help = "The file path of the decomposed hypotheses.",
    )
    parser.add_argument(
        "--NBC_file_path",
        type = str,
        default = "dataset/NBC",
        help = "The file path of the NBC features.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--C_M",
        type=int,
        default=28,
        help="Cost of Miss, we incur a cost when mistakenly classifying hallucinations as factual information.",
    )
    parser.add_argument(
        "--C_FA",
        type=int,
        default=96,
        help=" Cost of False Alarm, we incur a cost when mistakenly classifying factual information as hallucinations.",
    )
    parser.add_argument(
        "--C_Retrieve",
        type=int,
        default=1,
        help="Cost of retrieve an external evidence.",
    )
    parser.add_argument(
        "--P_0",
        type=float,
        default=0.5,
        help="The initial probability of hallucination.",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=400,
        help="The length of segment, used in split text.",
    )
    parser.add_argument(
        "--overlap_length",
        type=int,
        default=100,
        help="The length of overlap between segments, used in split text",
    )

    args = parser.parse_args()

    return args


def min_cost(p, C_M, C_FA):
    # Rstop(n) = min((1 − π1(n))*C_M, (1 − π0(n))*C_FA) 
    # For R_continue, p is the expectation of time n+1
    
    return min((1-p)*C_M,p*C_FA)

def cal_En_plus1(neg_features, pos_features, P_n):
    P_nplus1 = [0] * 10
    
    for i in range(10):
        P_nplus1_given_1 = pos_features[i] / sum(pos_features)
        P_nplus1_given_0 = neg_features[i] / sum(neg_features)
        
        # π1(n+ 1) = π1(n)P(fn+1|θ1)/((1 − π1(n))P(fn+1|θ0) + π1(n)P(fn+1|θ1))
        P_nplus1[i] = P_n*P_nplus1_given_1/((1-P_n)*P_nplus1_given_0+P_n*P_nplus1_given_1)
        
    E_nplus1 = sum(P_nplus1)/len(P_nplus1)
    #print(E_nplus1)
    
    return E_nplus1

def main():
    args = parse_args()

    dataset = get_data(args.test_file_path)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("loading tokenizer...", flush = True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("loading model...", flush = True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    
    NBC_file_path = [args.NBC_file_path + "/NBC_positive.json", args.NBC_file_path + "/NBC_negative.json"]
    #print(NBC_file_path)
    NBC_features = get_NBC_features(model, tokenizer, NBC_file_path)
    neg_features = NBC_features['neg_features']
    pos_features = NBC_features['pos_features']
    
    #Laplace smoothing
    neg_features = [x + 1 for x in neg_features]
    pos_features = [x + 1 for x in pos_features]
    
    print(neg_features, pos_features, flush=True)
    
    print("evaluting...", flush = True)
    evaluate(
        dataset = dataset,
        tokenizer = tokenizer,
        model = model,
        C_M = args.C_M,
        C_FA = args.C_FA,
        C_retrieve = args.C_Retrieve,
        P0 = args.P_0,
        neg_features = neg_features,
        pos_features = pos_features,
        args = args
    )

def evaluate(dataset, tokenizer, model, C_M, C_FA, C_retrieve, P0, neg_features, pos_features, args):
    passage_index = 0
    wrong_num = 0
    total_num = 0
    
    sentence_golden_label_list = []
    sentence_test_list = []
    passage_golden_label_list = []
    passage_test_label_list = []
    
    sentence_search_time_list = []
    hypothesis_search_time_list = []
    
    for passage in tqdm(dataset):
        # print(passage_index,flush=True)
        passage_prob_golden = []
        passage_test_label = []
        # print(passage['gpt3_text'])
        sentence_index = 0
        
        for sentence in passage['gpt3_sentences']:
            sentence_instance = {}
            sentence_instance['sentence'] = sentence
            golden_label = passage['annotation'][sentence_index]
            
            # We will make the major_ Inaccurate and inaccurate are both considered inaccurate
            if golden_label == 'accurate':
                golden_label = 1
            else:
                golden_label =0
                
            passage_prob_golden.append(golden_label)
            sentence_instance['label'] = golden_label
            
            web_file_name = args.webpage_file_path + '/'+ str(passage_index) +'_' + str(sentence_index) + '.json'
            hypothesis_file_name = args.decomposed_file_path + '/decomposed_' + str(passage_index) + '_' + str(sentence_index) + '.txt'
            
            hypothesis_list = get_data(hypothesis_file_name)
            sentence_web = get_data(web_file_name)
            
            hypothesis_P_list = []
            hypothesis_search_time = []
            
            for hypothesis in hypothesis_list:
                hypothesis_web = sentence_web[hypothesis]
                
                # initial P = P0
                P = P0
                search_time = 0
                stop_cost = min_cost(P, C_M, C_FA)
                search_cost = C_retrieve + min_cost(cal_En_plus1(neg_features, pos_features, P), C_M, C_FA)
                
                for web in hypothesis_web:
                    if stop_cost > search_cost:
                        search_time += 1
                        
                        # We calculate the entailment score between each text span and the corresponding subclaim, and select the highest entailment score as the entailment score for the original document:
                        segment_list = split_text(web['page_content'],args.segment_length,args.overlap_length)
                        Entailment_prob = 0
                        for segment in segment_list:
                            new_Entailment_prob, _ = get_entailment_score(
                                premise = segment,
                                hypothesis = hypothesis,
                                tokenizer = tokenizer,
                                model = model,
                            )
                            if new_Entailment_prob > Entailment_prob:
                                Entailment_prob = new_Entailment_prob
                                
                        P_nplus1_given_1 = pos_features[int((Entailment_prob-0.1)/10)] / sum(pos_features)
                        P_nplus1_given_0 = neg_features[int((Entailment_prob-0.1)/10)] / sum(neg_features)
                        P = P*P_nplus1_given_1/((1-P)*P_nplus1_given_0+P*P_nplus1_given_1)  
                        #print(P,flush=True) 
                           
                        stop_cost = min_cost(P, C_M, C_FA)
                        search_cost = C_retrieve + min_cost(cal_En_plus1(neg_features, pos_features, P),C_M, C_FA)
                        
                hypothesis_P_list.append(P)
                hypothesis_search_time.append(search_time)
                sentence_search_time_list.append(search_time)

            hypothesis_list = []
            final_P = min(hypothesis_P_list)
            passage_test_label.append(final_P)
            
            final_search_time = sum(hypothesis_search_time)
            hypothesis_search_time_list.append(final_search_time)
            
            sentence_golden_label_list.append(golden_label)
            sentence_test_list.append(final_P)
            sentence_index+=1
            
            if ((1-final_P)*C_M) < (final_P*C_FA):
                predict_label = 1
            else:
                predict_label = 0 
                
            if predict_label != golden_label:
                wrong_num+=1
                print(wrong_num, total_num, flush=True)
                
            total_num+=1
            
        passage_index+=1
        passage_golden_label_list.append(sum(passage_prob_golden)/len(passage_prob_golden))
        passage_test_label_list.append(sum(passage_test_label)/len(passage_test_label))

    y_test = [1-i for i in sentence_test_list]
    y_true = [1-i for i in sentence_golden_label_list]
    precision, recall, thresholds = precision_recall_curve(y_true, y_test)
    # Use AUC function to calculate the area under the curve of precision recall curve
    Non_fact_auc_precision_recall = auc(recall, precision)
    print("acc: ",1-wrong_num/total_num)
    print("Non_fact_auc_precision_recall: ",Non_fact_auc_precision_recall)

    precision, recall, thresholds = precision_recall_curve(sentence_golden_label_list, sentence_test_list)
    # Use AUC function to calculate the area under the curve of precision recall curve
    fact_auc_precision_recall = auc(recall, precision)
    
    print("fact_auc_precision_recall: ",fact_auc_precision_recall)
    print("avg_sentence_search_time:", sum(hypothesis_search_time_list)/len(hypothesis_search_time_list))
    print("pearson: ",scipy.stats.pearsonr(passage_golden_label_list, passage_test_label_list)[0])
    print("Spearman: ",scipy.stats.spearmanr(passage_golden_label_list, passage_test_label_list)[0])
    print("hypothesis_avg_search_time:", sum(sentence_search_time_list)/len(sentence_search_time_list)) 
    
if __name__ == "__main__":
    main()   
    