import os
import json
import subprocess
import sentencepiece as spm
import torch
from pathlib import Path

# 1. Data Preparation (Demo: create small JSONL file)
def prepare_data():
    os.makedirs('data', exist_ok=True)
    data = [
        {"en": "The intricacies of quantum entanglement challenge our classical understanding of reality.",
         "hi": "क्वांटम उलझाव की जटिलताएँ हमारी पारंपरिक वास्तविकता की समझ को चुनौती देती हैं।",
         "mr": "क्वांटम गुंतागुंतीचे बारकावे आपल्या पारंपरिक वास्तवाच्या समजुतीला आव्हान देतात."},
        {"en": "Despite the overwhelming odds, the expedition persevered through the relentless blizzard.",
         "hi": "भारी बाधाओं के बावजूद, अभियान ने लगातार बर्फीले तूफान में भी दृढ़ता दिखाई।",
         "mr": "प्रचंड अडचणी असूनही, मोहिमेने सततच्या हिमवादळातही चिकाटी ठेवली."},
        {"en": "Her eloquence in articulating complex philosophical concepts left the audience spellbound.",
         "hi": "जटिल दार्शनिक अवधारणाओं को व्यक्त करने में उसकी वाक्पटुता ने श्रोताओं को मंत्रमुग्ध कर दिया।",
         "mr": "गूढ तत्त्वज्ञानाच्या संकल्पना स्पष्टपणे मांडण्यात तिच्या वक्तृत्वाने प्रेक्षक मंत्रमुग्ध झाले."}
    ]
    with open('data/translation_benchmark_en_hi_mr.jsonl', 'w', encoding='utf-8') as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print('Data prepared.')

# 2. Train SentencePiece Tokenizer (Demo: train on small data)
def train_tokenizer():
    # Combine all sentences
    with open('data/all_texts.txt', 'w', encoding='utf-8') as out:
        with open('data/translation_benchmark_en_hi_mr.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                for lang in ['en', 'hi', 'mr']:
                    out.write(ex[lang] + '\n')
    spm.SentencePieceTrainer.Train('--input=data/all_texts.txt --model_prefix=data/spm --vocab_size=100 --character_coverage=1.0')
    print('Tokenizer trained.')

# 3. Dummy Model Training (for demo)
def train_model():
    from smart_transformer import SmartTransformer, AdaptiveConfig
    config = AdaptiveConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=64,
        use_adaptive_attention=True,
    )
    model = SmartTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    labels = input_ids.clone()
    shifted_input_ids = input_ids[:, :-1]
    shifted_labels = labels[:, 1:]
    outputs = model(input_ids=shifted_input_ids)
    logits = outputs['logits']
    if logits.size(1) > shifted_labels.size(1):
        logits = logits[:, :shifted_labels.size(1), :]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), shifted_labels.reshape(-1))
    loss.backward()
    optimizer.step()
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, 'smart_transformer_example.pt')
    print('Dummy model trained and exported.')

# 4. Run Benchmark
def run_benchmark():
    subprocess.run(['python', 'translation_benchmark.py'], check=True)
    print('Benchmark completed.')

# 5. (Optional) Update Dashboard (already reads results file)
def main():
    prepare_data()
    train_tokenizer()
    train_model()
    run_benchmark()
    print('All steps complete. Check results_dashboard.html for updated results.')

if __name__ == '__main__':
    main() 