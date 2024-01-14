import json
import torch

def get_data(file_path):
    """
    Read data from file.

    Args:
        file_path (str): The path of dataset. Txt file for decomposed sentences and json for NBC and webpages.

    Returns:
        Dict / List[str]: Data.
    """
    try:
        file_extension = file_path.split('.')[-1]

        if file_extension == 'txt':
            data = []
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
                lines = txt_file.readlines()     
                for line in lines:
                    if line.isspace():
                        continue
                    else:
                        line = line.replace("\n","")
                        data.append(line)
                #print(f"Text content from {file_path}:\n{content}")

        elif file_extension == 'json':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as json_file:
                data = json.load(json_file)
                #print(f"JSON data from {file_path}:\n{json.dumps(data, indent=2)}")
            
        return data
    except Exception as e:
        print(f"Error reading file: {e}")



def get_entailment_score(premise, hypothesis, tokenizer, model):
    """
    Get entailment score for the given premise and hypothesis.

    Args:
        premise (str): The premise text.
        hypothesis (str): The hypothesis text.
        tokenizer: Tokenizer object.
        model: Entailment model.

    Returns:
        Tuple[float, float]: Entailment probability and disentailment probability.
    """
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    output = model(inputs["input_ids"].to(device))
    
    probabilities = torch.softmax(output["logits"][0] / 5, -1).tolist()
    label_names = ["entailment", "neutral", "not_entailment"]
    # Create a dictionary with rounded probabilities
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(probabilities, label_names)}
    
    # Extract entailment and disentailment probabilities
    entailment_prob = prediction['entailment']
    disentailment_prob = prediction['not_entailment'] + prediction['neutral']
    
    return entailment_prob, disentailment_prob


def split_text(text, segment_length, overlap_length):
    """
    Split text into segment.
    """
    segments = []
    start = 0
    end = segment_length
    text_list = text.split()[0:4000]
    while start < len(text_list):
        if end >= len(text_list):
            segment = text_list[-segment_length:]
        else:
            segment = text_list[start:end]
        segments.append(" ".join(segment))
        start += segment_length - overlap_length
        end = start + segment_length
        
    return segments