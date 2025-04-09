from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model, BitsAndBytesConfig
import torch
import gc


def load_model(model_name, device="cuda", quantize=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        torch_dtype=torch.float16
        ) if quantize else None

    if "hubert" in model_name.lower():
        model = HubertModel.from_pretrained(model_name, quantization_config=bnb_config)
    else:  # For both wav2vec2 and xlsr models
        model = Wav2Vec2Model.from_pretrained(model_name, quantization_config=bnb_config)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model.to(device)
    return model, feature_extractor

def collate_fn(batch, multilingual=False):
    batch.sort(key=lambda x: x['input_values'].size(0), reverse=True)

    input_values = [item['input_values'] for item in batch]
    texts = [item['text'] for item in batch]

    input_lengths = torch.tensor([x.size(0) for x in input_values])

    max_length = input_lengths[0].item()
    padded_inputs = torch.zeros(len(batch), max_length)

    for i, (input_value, length) in enumerate(zip(input_values, input_lengths)):
        padded_inputs[i, :length] = input_value

    result = {
        'input_values': padded_inputs,
        'input_lengths': input_lengths,
        'text': texts
    }

    if multilingual:
        languages = [item['language'] for item in batch]
        lang_ids = torch.tensor([item['lang_id'] for item in batch])
        result['languages'] = languages
        result['lang_ids'] = lang_ids

    return result

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
