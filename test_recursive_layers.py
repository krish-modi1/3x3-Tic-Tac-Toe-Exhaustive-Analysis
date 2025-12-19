#!/usr/bin/env python3
"""
Test Recursive Layer Relationships - Position 4

Professor's Hypothesis:
  Layer 3 ≈ baseline + w1 × Layer_2_pattern
  Layer 4 ≈ baseline + w2 × Layer_3_pattern

If patterns are self-similar, we should find consistent weights!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def build_hierarchical_order(pdf):
    """Build hierarchical ordering"""
    plot_data = []

    l1 = pdf[pdf['layer'] == 1]
    if len(l1) > 0:
        plot_data.append({'layer': 1, 'state': l1.iloc[0]['state'], 'px': l1.iloc[0]['P(X wins)'], 'x_order': 0})

    l2 = pdf[pdf['layer'] == 2].sort_values('P(X wins)')
    for i, (_, r) in enumerate(l2.iterrows()):
        plot_data.append({'layer': 2, 'state': r['state'], 'px': r['P(X wins)'], 'x_order': i})

    for layer in range(3, 10):
        ld = pdf[pdf['layer'] == layer]
        if len(ld) == 0:
            break
        parents = [d['state'] for d in plot_data if d['layer'] == layer-1]
        grouped = defaultdict(list)
        for _, r in ld.iterrows():
            if r['parent_state'] in parents:
                grouped[r['parent_state']].append({'state': r['state'], 'px': r['P(X wins)']})
        x_ord = 0
        for p in parents:
            children = grouped.get(p, [])
            children.sort(key=lambda x: x['px'])
            for c in children:
                plot_data.append({'layer': layer, 'state': c['state'], 'px': c['px'], 'x_order': x_ord})
                x_ord += 1

    return pd.DataFrame(plot_data)


def normalize_pattern(values):
    """Normalize to [0, 1] range for comparison"""
    if len(values) == 0 or np.std(values) == 0:
        return values
    return (values - np.min(values)) / (np.max(values) - np.min(values))


def resample_pattern(pattern, target_length):
    """Resample pattern to match target length using interpolation"""
    if len(pattern) == target_length:
        return pattern

    # Linear interpolation
    x_old = np.linspace(0, 1, len(pattern))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, pattern)


def find_recursive_weight(child_layer, parent_layer):
    """
    Find weight w such that: child_layer ≈ baseline + w × parent_pattern

    Returns: w, baseline, residuals, mae
    """
    # Normalize both patterns
    parent_norm = normalize_pattern(parent_layer)
    child_norm = normalize_pattern(child_layer)

    # Resample parent to match child length
    parent_resampled = resample_pattern(parent_norm, len(child_norm))

    # Find optimal w and baseline using least squares
    # child = baseline + w * parent
    # Solve: [1, parent] * [baseline, w]^T = child

    X = np.column_stack([np.ones(len(child_norm)), parent_resampled])
    w_opt, residuals, _, _ = np.linalg.lstsq(X, child_norm, rcond=None)

    baseline, w = w_opt[0], w_opt[1]

    # Compute predicted and error
    predicted = baseline + w * parent_resampled
    mae = np.mean(np.abs(child_norm - predicted))

    return w, baseline, predicted, mae


print("="*70)
print("TESTING RECURSIVE LAYER RELATIONSHIPS - POSITION 4")
print("="*70)
print()

# Load data
df = pd.read_csv('position_probabilities/position_4_probabilities.csv')
pdf = df[df['state'].str[4] == 'X'].copy()
ordered_df = build_hierarchical_order(pdf)

# Extract layers
layers = {}
for layer in range(2, 10):
    ld = ordered_df[ordered_df['layer'] == layer]
    if len(ld) > 0:
        layers[layer] = ld['px'].values

print(f"Loaded {len(layers)} layers\n")

# Test recursive relationships
results = []

print("HYPOTHESIS TEST: Layer N ≈ baseline + w × Layer(N-1)_pattern")
print("-"*70)

for child_layer_num in range(3, 9):
    if child_layer_num not in layers or child_layer_num - 1 not in layers:
        continue

    parent = layers[child_layer_num - 1]
    child = layers[child_layer_num]

    w, baseline, predicted, mae = find_recursive_weight(child, parent)

    results.append({
        'parent': child_layer_num - 1,
        'child': child_layer_num,
        'weight': w,
        'baseline': baseline,
        'mae': mae,
        'parent_len': len(parent),
        'child_len': len(child)
    })

    print(f"Layer {child_layer_num} ← Layer {child_layer_num-1}:")
    print(f"  Weight:   {w:+.4f}")
    print(f"  Baseline: {baseline:.4f}")
    print(f"  MAE:      {mae:.4f} ({mae*100:.1f}% error)")
    print(f"  Sizes:    {len(parent)} → {len(child)} states")
    print()

# Summary statistics
weights = [r['weight'] for r in results]
maes = [r['mae'] for r in results]

print("="*70)
print("SUMMARY")
print("="*70)
print(f"Average Weight:  {np.mean(weights):.4f} ± {np.std(weights):.4f}")
print(f"Weight Range:    [{np.min(weights):.4f}, {np.max(weights):.4f}]")
print(f"Average MAE:     {np.mean(maes):.4f} ({np.mean(maes)*100:.1f}% error)")
print()

if np.std(weights) < 0.3:
    print("✓ CONSISTENT WEIGHTS - Self-similar pattern detected!")
else:
    print("✗ INCONSISTENT WEIGHTS - No clear self-similarity")

print()

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Recursive Layer Relationships - Position 4 (Center)', fontsize=14, weight='bold')

for idx, result in enumerate(results[:6]):
    ax = axes[idx // 3, idx % 3]

    parent_num = result['parent']
    child_num = result['child']

    parent = layers[parent_num]
    child = layers[child_num]

    # Get prediction
    w, baseline, predicted, mae = find_recursive_weight(child, parent)

    # Normalize for plotting
    parent_norm = normalize_pattern(parent)
    child_norm = normalize_pattern(child)
    parent_resampled = resample_pattern(parent_norm, len(child_norm))

    x = np.arange(len(child_norm))

    ax.plot(x, child_norm, 'b-', label=f'Layer {child_num} (actual)', lw=1.5, alpha=0.7)
    ax.plot(x, predicted, 'r--', label=f'Predicted (w={w:.2f})', lw=1.5, alpha=0.7)

    ax.set_title(f'Layer {child_num} ← {parent_num} (MAE={mae:.3f})', fontsize=11)
    ax.set_xlabel('State Index (Normalized)', fontsize=9)
    ax.set_ylabel('P(X wins) [0-1 normalized]', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Add weight annotation
    ax.text(0.02, 0.98, f'w = {w:.3f}\nbaseline = {baseline:.3f}', 
           transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('position_4_recursive_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: position_4_recursive_analysis.png\n")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('recursive_weights.csv', index=False)
print("Saved: recursive_weights.csv")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
If the weights are consistent (std < 0.3):
  → Each layer is a scaled/shifted version of previous layer
  → Self-similar fractal-like structure
  → Potential for recursive prediction formula

If weights vary significantly:
  → Each layer has unique structure
  → No simple recursive relationship
  → Need layer-specific models

MAE < 0.15: Good fit (recursive relationship exists)
MAE > 0.25: Poor fit (no clear recursion)
""")
