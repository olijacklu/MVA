import editdistance


def calculate_cer(reference, predicted):
    ref_len = len(reference)
    if ref_len == 0:
        return 1.0 if len(predicted) > 0 else 0.0
    return editdistance.eval(reference, predicted) / ref_len
