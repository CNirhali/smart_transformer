import subprocess
import itertools
import json
import os

# Define grid of hyperparameters
param_grid = {
    'hidden_size': [128, 256],
    'num_layers': [2, 4],
    'batch_size': [2, 4],
    'learning_rate': [1e-4, 5e-5],
    'use_adaptive_attention': [True, False],
}

# Create all combinations
keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []

for i, exp in enumerate(experiments):
    print(f"\n=== Running experiment {i+1}/{len(experiments)}: {exp} ===")
    args = [
        f"--hidden_size={exp['hidden_size']}",
        f"--num_layers={exp['num_layers']}",
        f"--batch_size={exp['batch_size']}",
        f"--learning_rate={exp['learning_rate']}",
        f"--use_adaptive_attention={int(exp['use_adaptive_attention'])}",
    ]
    # Run the script
    result = subprocess.run([
        'python', 'examples/advanced_training.py', *args
    ], capture_output=True, text=True)
    # Save output
    out_dir = f"results/exp_{i+1}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'stdout.txt'), 'w') as f:
        f.write(result.stdout)
    with open(os.path.join(out_dir, 'stderr.txt'), 'w') as f:
        f.write(result.stderr)
    # Try to extract metrics from output (simple parse)
    metrics = {}
    for line in result.stdout.splitlines():
        if 'Loss:' in line or 'Accuracy:' in line or 'F1' in line or 'BLEU' in line or 'ROUGE' in line:
            metrics[line.strip()] = True
    results.append({'config': exp, 'metrics': metrics})

# Print summary
print("\n=== Experiment Summary ===")
for r in results:
    print(r['config'])
    for m in r['metrics']:
        print(f"  {m}")
    print() 