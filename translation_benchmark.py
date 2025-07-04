import json
import time
from pathlib import Path
from collections import defaultdict
import torch
from smart_transformer import SmartTransformer, AdaptiveConfig
from smart_transformer.core import AdaptiveConfig as CoreAdaptiveConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

DATA_PATH = Path('data/translation_benchmark_en_hi_mr.jsonl')
MODEL_PATH = Path('smart_transformer_example.pt')

# --- Dummy Tokenizer for Demo ---
def dummy_tokenizer(text):
    tokens = text.split()
    vocab = {w: i % 1000 for i, w in enumerate(set(tokens))}
    return [vocab[w] for w in tokens]

def dummy_detokenizer(token_ids):
    return ' '.join(str(i) for i in token_ids)

# --- Load Model ---
def load_model():
    torch.serialization.add_safe_globals([CoreAdaptiveConfig])
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = SmartTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

# --- Batch Translation ---
def smart_transformer_batch_translate(model, config, texts, target_lang):
    input_ids = [dummy_tokenizer(text) for text in texts]
    max_len = max(len(ids) for ids in input_ids)
    input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=30,
            do_sample=False,
            task_type='translation',
            domain='general',
            technique='standard',
        )
    preds = []
    for i, ids in enumerate(output_ids):
        gen_ids = ids.tolist()[len(input_ids[i]):]
        preds.append(dummy_detokenizer(gen_ids))
    return preds

# --- Evaluation ---
def evaluate(preds, refs):
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    for pred, ref in zip(preds, refs):
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
        scores = scorer.score(ref, pred)
        bleu_scores.append(bleu)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    return {
        'BLEU': sum(bleu_scores)/len(bleu_scores),
        'ROUGE-1': sum(rouge1_scores)/len(rouge1_scores),
        'ROUGE-L': sum(rougeL_scores)/len(rougeL_scores),
    }

def main():
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    model, config = load_model()
    results = defaultdict(dict)
    for lang in ['hi', 'mr']:
        refs = [ex[lang] for ex in data]
        en_texts = [ex['en'] for ex in data]
        start = time.time()
        preds = smart_transformer_batch_translate(model, config, en_texts, lang)
        elapsed = (time.time() - start) / len(data) * 1000  # ms/sentence
        metrics = evaluate(preds, refs)
        results[lang]['metrics'] = metrics
        results[lang]['speed_ms_per_sent'] = elapsed
    for lang in ['hi', 'mr']:
        print(f"\nResults for EN→{lang.upper()}:")
        for k, v in results[lang]['metrics'].items():
            print(f"  {k}: {v:.4f}")
        print(f"  Speed: {results[lang]['speed_ms_per_sent']:.2f} ms/sentence")
    with open('data/translation_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main() 