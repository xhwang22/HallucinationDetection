import json
import torch
from utils import get_entailment_score, get_data

def get_NBC_features(model, tokenizer, data_path):
    """
    Get NBC features which are discretized based on entailment scores.

    Args:
        model: Entailment model.
        tokenizer: Tokenizer object.
        data_path (list): Positive and negative data_path.

    Returns:
        Dict[list, list]: Positive feature list and Negative feature list.
    """
    
    pos_feature_list = [0] * 10
    neg_feature_list = [0] * 10
    
    pos_data = get_data(data_path[0])
    neg_data = get_data(data_path[1])
    
    for data in pos_data:
        premise = data['premise']
        hypothesis = data['hypothesis']
        score, _ = get_entailment_score(
            premise = premise,
            hypothesis = hypothesis,
            tokenizer = tokenizer,
            model = model,
        )
        # Discretization
        pos_feature_list[int(score/10)] += 1
    
    for data in neg_data:
        premise = data['premise']
        hypothesis = data['hypothesis']
        score, _ = get_entailment_score(
            premise = premise,
            hypothesis = hypothesis,
            tokenizer = tokenizer,
            model = model,
        )
        # Discretization
        neg_feature_list[int(score/10)] += 1 

    return {
        "pos_features": pos_feature_list,
        "neg_features": neg_feature_list,
    }
