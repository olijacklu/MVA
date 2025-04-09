import torch
from torch.utils.data import DataLoader

from data.preprocess import prepare_multilingual_data
from data.dataset import MLSuperbDataset
from models.utils import collate_fn
from evaluation.metrics import calculate_cer


def test_model(model, feature_extractor, datasets, char_mappings, model_type="monolingual", data_pair=None, lang_id=None, task="asr", num_samples=10, device="cuda"):
    model.eval()

    idx_to_char = char_mappings['idx_to_char']
    char_to_idx = char_mappings['char_to_idx']

    if model_type == "monolingual":
        test_samples = datasets[data_pair]['splits']['10min_test'][:num_samples]
        test_dataset = MLSuperbDataset(test_samples, feature_extractor)
        test_loader = DataLoader(test_dataset, batch_size=len(test_samples), collate_fn=collate_fn, num_workers=4, pin_memory=True)

    else: # multilingual
        if lang_id:
            test_samples = [sample for sample in prepare_multilingual_data(datasets, '10min_test')
                           if sample['language'] == lang_id][:num_samples]
        else:
            all_test_samples = prepare_multilingual_data(datasets, '10min_test')
            languages = set(sample['language'] for sample in all_test_samples)

            test_samples = []
            for language in languages:
                lang_samples = [s for s in all_test_samples if s['language'] == language]
                samples_to_add = lang_samples[:min(num_samples, len(lang_samples))]
                test_samples.extend(samples_to_add)

            print(f"Testing on {len(languages)} languages with {len(test_samples)} total samples")

        test_dataset = MLSuperbDataset(test_samples, feature_extractor, multilingual=(task != "asr"))
        test_loader = DataLoader(test_dataset, batch_size=len(test_samples),
                               collate_fn=lambda batch: collate_fn(batch, multilingual=(task != "asr")),
                               num_workers=4, pin_memory=True)

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch['input_values'].to(device)
            texts = batch['text']

            if task == "asr":
                logits = model(input_values)
                predictions = torch.argmax(logits, dim=-1)

                print(f"\nASR Test Examples on {data_pair if model_type == 'monolingual' else 'multilingual'}:")

                if model_type == "multilingual" and "languages" in batch:
                    languages = batch['languages']
                    results_by_lang = {}

                    for i, (pred, text, lang) in enumerate(zip(predictions, texts, languages)):
                        pred_indices = pred.cpu().numpy()

                        merged_indices = []
                        prev_idx = -1
                        for idx in pred_indices:
                            if idx != prev_idx and idx != 0:
                                merged_indices.append(idx)
                            prev_idx = idx

                        pred_text = ''.join([idx_to_char[idx] for idx in merged_indices if idx in idx_to_char])
                        cer = calculate_cer(text, pred_text)

                        if lang not in results_by_lang:
                            results_by_lang[lang] = []

                        results_by_lang[lang].append({
                            "index": i+1,
                            "reference": text,
                            "prediction": pred_text,
                            "cer": cer
                        })

                    all_cers = [r["cer"] for results in results_by_lang.values() for r in results]
                    overall_cer = sum(all_cers) / len(all_cers) if all_cers else 0
                    print(f"Overall CER: {overall_cer:.4f}")

                    for lang, results in sorted(results_by_lang.items()):
                        lang_cer = sum(r["cer"] for r in results) / len(results)
                        print(f"\n=== Language: {lang} ===")
                        print(f"Average CER: {lang_cer:.4f}")

                        for r in results:
                            print(f"Example {r['index']}:")
                            print(f"Reference: {r['reference']}")
                            print(f"Prediction: {r['prediction']}")
                            print(f"CER: {r['cer']:.4f}")
                else:
                    for i, (pred, text) in enumerate(zip(predictions, texts)):
                        pred_indices = pred.cpu().numpy()

                        merged_indices = []
                        prev_idx = -1
                        for idx in pred_indices:
                            if idx != prev_idx and idx != 0:
                                merged_indices.append(idx)
                            prev_idx = idx

                        pred_text = ''.join([idx_to_char[idx] for idx in merged_indices if idx in idx_to_char])
                        print(f"Example {i+1}:")
                        print(f"Reference: {text}")
                        print(f"Prediction: {pred_text}")
                        print(f"CER: {calculate_cer(text, pred_text):.4f}")

            elif task == "lid":
                outputs = model(input_values)
                lid_logits = outputs["lid_logits"]
                lid_predictions = torch.argmax(lid_logits, dim=1)

                languages = batch['languages']
                lang_ids = batch['lang_ids'].to(device)

                print(f"\nLID Test Examples:")

                results_by_lang = {}
                for i, (pred, lang, lang_id) in enumerate(zip(lid_predictions, languages, lang_ids)):
                    is_correct = pred == lang_id

                    if lang not in results_by_lang:
                        results_by_lang[lang] = []

                    results_by_lang[lang].append({
                        "index": i+1,
                        "true_lang": lang,
                        "pred_id": pred.item(),
                        "correct": is_correct
                    })

                all_results = [r["correct"] for results in results_by_lang.values() for r in results]
                overall_accuracy = sum(1 for r in all_results if r) / len(all_results) if all_results else 0
                print(f"Overall LID Accuracy: {overall_accuracy:.4f}")

                for lang, results in sorted(results_by_lang.items()):
                    lang_accuracy = sum(1 for r in results if r["correct"]) / len(results)
                    print(f"\n=== Language: {lang} ===")
                    print(f"Accuracy: {lang_accuracy:.4f}")

                    for r in results:
                        print(f"Example {r['index']}:")
                        print(f"True language: {r['true_lang']}")
                        print(f"Predicted language ID: {r['pred_id']}")
                        print(f"Correct: {r['correct']}")

            elif task == "asr+lid":
                outputs = model(input_values)
                asr_logits = outputs["asr_logits"]
                lid_logits = outputs["lid_logits"]

                asr_predictions = torch.argmax(asr_logits, dim=-1)
                lid_predictions = torch.argmax(lid_logits, dim=1)

                languages = batch['languages']
                lang_ids = batch['lang_ids'].to(device)

                print(f"\nJoint ASR+LID Test Examples:")

                results_by_lang = {}

                for i, (asr_pred, lid_pred, text, lang, lang_id) in enumerate(
                        zip(asr_predictions, lid_predictions, texts, languages, lang_ids)):

                    pred_indices = asr_pred.cpu().numpy()
                    merged_indices = []
                    prev_idx = -1
                    for idx in pred_indices:
                        if idx != prev_idx and idx != 0:
                            merged_indices.append(idx)
                        prev_idx = idx

                    pred_text = ''.join([idx_to_char[idx] for idx in merged_indices if idx in idx_to_char])
                    cer = calculate_cer(text, pred_text)
                    lid_correct = lid_pred == lang_id

                    if lang not in results_by_lang:
                        results_by_lang[lang] = []

                    results_by_lang[lang].append({
                        "index": i+1,
                        "reference": text,
                        "prediction": pred_text,
                        "cer": cer,
                        "lid_pred": lid_pred.item(),
                        "lid_correct": lid_correct
                    })

                all_cers = [r["cer"] for results in results_by_lang.values() for r in results]
                all_lid_correct = [r["lid_correct"] for results in results_by_lang.values() for r in results]
                overall_cer = sum(all_cers) / len(all_cers) if all_cers else 0
                overall_lid_accuracy = sum(1 for r in all_lid_correct if r) / len(all_lid_correct) if all_lid_correct else 0

                print(f"Overall ASR CER: {overall_cer:.4f}")
                print(f"Overall LID Accuracy: {overall_lid_accuracy:.4f}")

                for lang, results in sorted(results_by_lang.items()):
                    lang_cer = sum(r["cer"] for r in results) / len(results)
                    lid_accuracy = sum(1 for r in results if r["lid_correct"]) / len(results)
                    print(f"\n=== Language: {lang} ===")
                    print(f"Average CER: {lang_cer:.4f}")
                    print(f"LID Accuracy: {lid_accuracy:.4f}")

                    for r in results:
                        print(f"Example {r['index']}:")
                        print(f"Reference: {r['reference']}")
                        print(f"ASR Prediction: {r['prediction']}")
                        print(f"ASR CER: {r['cer']:.4f}")
                        print(f"LID Prediction: {r['lid_pred']}")
                        print(f"LID Correct: {r['lid_correct']}")
