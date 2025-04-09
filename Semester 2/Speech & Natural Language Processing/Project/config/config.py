import torch

# Dictionary mapping language codes to datasets
TRAIN_PAIRS = {
    'eng1': 'eng_mls',
    'eng2': 'eng_nchlt',
    'eng3': 'eng_voxpopuli',
    'fra1': 'fra_mls',
    'fra2': 'fra_voxpopuli',
    'deu1': 'deu_swc',
    'deu2': 'deu_voxpopuli',
    'rus': 'rus_M-AILABS',
    'swa': 'swa_ALFFA',
    'swe': 'swe_NST',
    'xty': 'xty_mexico-el'
}

TORCH_DEFAULT_TYPE = torch.float16
