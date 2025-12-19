#!/usr/bin/env python3
"""
Fourier Branch Analysis - All 9 Opening Positions

This script analyzes all 9 tic-tac-toe opening positions using:
  1. Hierarchical parent-child ordering
  2. Unconstrained feature weights (allows negative)
  3. Fourier approximation on ordered residuals

Outputs:
  - position_N_fourier_branch.png for each position (N=0-8)
  - fourier_branch_summary.json with all learned parameters
  - fourier_branch_results.txt with summary statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import sys

# ═══════════════════════════════════════════════════════════════════════
# HIERARCHICAL ORDERING
# ═══════════════════════════════════════════════════════════════════════

def build_hierarchical_order(pdf):
    """
    Build hierarchically ordered data using parent-child relationships.

    Layer 1: Single opening state
    Layer 2: Sorted by P(X wins) ascending
    Layer 3+: Grouped by parent, sorted within group by P(X wins)
    """
    plot_data = []

    # Layer 1
    l1 = pdf[pdf['layer'] == 1]
    if len(l1) > 0:
        plot_data.append({
            'layer': 1,
            'state': l1.iloc[0]['state'],
            'px': l1.iloc[0]['P(X wins)'],
            'x_order': 0
        })

    # Layer 2: Sort by probability
    l2 = pdf[pdf['layer'] == 2].sort_values('P(X wins)')
    for i, (_, r) in enumerate(l2.iterrows()):
        plot_data.append({
            'layer': 2,
            'state': r['state'],
            'px': r['P(X wins)'],
            'x_order': i
        })

    # Layer 3+: Group by parent
    for layer in range(3, 10):
        ld = pdf[pdf['layer'] == layer]
        if len(ld) == 0:
            break

        # Get parent states in order
        parents = [d['state'] for d in plot_data if d['layer'] == layer-1]

        # Group children by parent
        grouped = defaultdict(list)
        for _, r in ld.iterrows():
            if r['parent_state'] in parents:
                grouped[r['parent_state']].append({
                    'state': r['state'],
                    'px': r['P(X wins)']
                })

        # Add children in order
        x_ord = 0
        for p in parents:
            children = grouped.get(p, [])
            children.sort(key=lambda x: x['px'])
            for c in children:
                plot_data.append({
                    'layer': layer,
                    'state': c['state'],
                    'px': c['px'],
                    'x_order': x_ord
                })
                x_ord += 1

    return pd.DataFrame(plot_data)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_features(board_str):
    """Extract 5 game-theoretic features from board state"""
    board = [1 if c=='X' else -1 if c=='O' else 0 for c in board_str]

    # Center control
    center = (board[4] + 1) / 2

    # Corner control
    corners = sum(1 for i in [0,2,6,8] if board[i]==1) - sum(1 for i in [0,2,6,8] if board[i]==-1)
    corners_norm = (corners + 4) / 8

    # Edge control
    edges = sum(1 for i in [1,3,5,7] if board[i]==1) - sum(1 for i in [1,3,5,7] if board[i]==-1)
    edges_norm = (edges + 4) / 8

    # Threats (2-in-a-row)
    lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    threats = 0
    for line in lines:
        xc = sum(1 for i in line if board[i]==1)
        oc = sum(1 for i in line if board[i]==-1)
        if xc==2 and oc==0:
            threats += 1
        if oc==2 and xc==0:
            threats -= 1
    threats_norm = (threats + 8) / 16

    # Density
    density = sum(1 for x in board if x!=0) / 9

    return [center, corners_norm, edges_norm, threats_norm, density]


# ═══════════════════════════════════════════════════════════════════════
# MODEL LEARNING
# ═══════════════════════════════════════════════════════════════════════

def learn_model(layer_data):
    """
    Learn Fourier + feature model for one layer.

    Uses UNCONSTRAINED least squares (allows negative weights).
    """
    if len(layer_data) < 5:
        return None

    # Extract features and targets
    X = np.array([extract_features(r['state']) for _, r in layer_data.iterrows()])
    y = np.array([r['px'] for _, r in layer_data.iterrows()])

    # Baseline
    baseline = np.mean(y)
    y_res = y - baseline

    # Learn feature weights (unconstrained)
    w, _, _, _ = np.linalg.lstsq(X, y_res, rcond=None)

    # Fourier on residuals
    fourier_res = y_res - X @ w

    if len(fourier_res) > 10:
        fft = np.fft.fft(fourier_res)
        freqs = np.fft.fftfreq(len(fourier_res))
        mag = np.abs(fft[1:len(fft)//2])

        if len(mag) > 0:
            idx = np.argmax(mag) + 1
            amp = np.abs(fft[idx]) / len(fourier_res) * 2
            phase = np.angle(fft[idx])
            period = 1.0 / abs(freqs[idx]) if freqs[idx] != 0 else len(fourier_res)
        else:
            amp, period, phase = 0, len(fourier_res), 0
    else:
        amp, period, phase = 0, len(fourier_res), 0

    return {
        'baseline': baseline,
        'weights': w,
        'amplitude': amp,
        'period': period,
        'phase': phase
    }


def predict(board_str, model, idx, n):
    """Predict probability using learned model"""
    feat = np.array(extract_features(board_str))
    prob = model['baseline'] + np.dot(feat, model['weights'])
    prob += model['amplitude'] * np.cos(2*np.pi*idx/max(n-1,1)*model['period'] + model['phase'])
    return max(0, min(1, prob))


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_position(pos):
    """Analyze one opening position"""
    csv_file = f'position_probabilities/position_{pos}_probabilities.csv'

    if not os.path.exists(csv_file):
        print(f"  Warning: {csv_file} not found")
        return None

    # Load and filter
    df = pd.read_csv(csv_file)
    pdf = df[df['state'].str[pos] == 'X'].copy()

    # Build hierarchical order
    ordered_df = build_hierarchical_order(pdf)

    # Learn models per layer
    models, preds, acts = {}, {}, {}

    for layer in sorted(ordered_df['layer'].unique()):
        if layer <= 1:
            continue

        ld = ordered_df[ordered_df['layer'] == layer].reset_index(drop=True)
        model = learn_model(ld)

        if model:
            models[layer] = model

            # Generate predictions
            p = [predict(r['state'], model, i, len(ld)) for i, (_, r) in enumerate(ld.iterrows())]
            a = [r['px'] for _, r in ld.iterrows()]

            preds[layer] = p
            acts[layer] = a

    return {
        'position': pos,
        'models': models,
        'predictions': preds,
        'actuals': acts,
        'n_states': len(pdf)
    }


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def plot_position(result, output_file):
    """Create 3x3 plot for one position"""
    pos = result['position']
    models = result['models']
    preds = result['predictions']
    acts = result['actuals']

    if not models:
        print(f"  No models to plot for position {pos}")
        return

    # Position type
    pos_type = {0:'corner', 1:'edge', 2:'corner', 3:'edge', 4:'center', 
                5:'edge', 6:'corner', 7:'edge', 8:'corner'}[pos]

    # Create figure
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                         left=0.08, right=0.92, top=0.94, bottom=0.06)

    fig.suptitle(f'Position {pos} ({pos_type.title()}): Fourier Branch Analysis', 
                fontsize=16, weight='bold')

    layers = sorted(models.keys())

    for i, layer in enumerate(layers[:9]):
        ax = fig.add_subplot(gs[i//3, i%3])

        actual = acts[layer]
        pred = preds[layer]
        x = range(len(actual))

        # Plot
        ax.plot(x, actual, 'b-', lw=1.8, alpha=0.75, label='Actual', marker='o', ms=2.5)
        ax.plot(x, pred, 'r--', lw=1.8, alpha=0.75, label='Fourier+Features', marker='s', ms=1.5)

        # Metrics
        mae = np.mean(np.abs(np.array(actual) - np.array(pred)))
        rmse = np.sqrt(np.mean((np.array(actual) - np.array(pred))**2))

        ax.set_title(f'Layer {layer} ({len(actual)} states) - MAE={mae:.3f}, RMSE={rmse:.3f}', 
                    fontsize=10, pad=8)
        ax.set_xlabel('State Index (Hierarchical Order)', fontsize=9)
        ax.set_ylabel('P(X wins)', fontsize=9)

        # Legend outside
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1), 
                 framealpha=0.9, edgecolor='gray')

        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)

        # Stats box
        stats = f'Min: {min(actual):.3f}\nMax: {max(actual):.3f}\nMean: {np.mean(actual):.3f}'
        ax.text(0.98, 0.02, stats, transform=ax.transAxes, fontsize=7, 
               va='bottom', ha='right',
               bbox=dict(boxstyle='round', fc='wheat', alpha=0.7, pad=0.4))

    # Fill empty subplots
    for i in range(len(layers), 9):
        ax = fig.add_subplot(gs[i//3, i%3])
        ax.text(0.5, 0.5, f'Layer {i+2}: No data', ha='center', va='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "="*76 + "╗")
    print("║" + " "*20 + "FOURIER BRANCH ANALYSIS - ALL 9 POSITIONS" + " "*16 + "║")
    print("╚" + "="*76 + "╝")
    print()

    all_results = {}
    summary_stats = []

    # Process each position
    for pos in range(9):
        print(f"Position {pos}...", end=' ')
        sys.stdout.flush()

        result = analyze_position(pos)

        if result is None:
            print("SKIPPED (no data)")
            continue

        all_results[pos] = result

        # Plot
        output_file = f'position_{pos}_fourier_branch.png'
        plot_position(result, output_file)

        # Calculate statistics
        maes = [np.mean(np.abs(np.array(result['actuals'][l]) - np.array(result['predictions'][l]))) 
                for l in result['models']]
        rmses = [np.sqrt(np.mean((np.array(result['actuals'][l]) - np.array(result['predictions'][l]))**2))
                 for l in result['models']]

        avg_mae = np.mean(maes) if maes else 0
        avg_rmse = np.mean(rmses) if rmses else 0

        summary_stats.append({
            'position': pos,
            'n_layers': len(result['models']),
            'n_states': result['n_states'],
            'mae': avg_mae,
            'rmse': avg_rmse
        })

        print(f"✓ MAE={avg_mae:.4f}, {len(result['models'])} layers, saved {output_file}")

    # Save parameters
    print("\nSaving learned parameters...")
    params = {}
    for pos, result in all_results.items():
        params[f'position_{pos}'] = {}
        for layer, model in result['models'].items():
            params[f'position_{pos}'][f'layer_{layer}'] = {
                'baseline': float(model['baseline']),
                'weights': [float(w) for w in model['weights']],
                'fourier': {
                    'amplitude': float(model['amplitude']),
                    'period': float(model['period']),
                    'phase': float(model['phase'])
                }
            }

    with open('fourier_branch_summary.json', 'w') as f:
        json.dump(params, f, indent=2)
    print("  ✓ Saved fourier_branch_summary.json")

    # Summary statistics
    print("\n" + "="*78)
    print("SUMMARY STATISTICS")
    print("="*78)
    print(f"{'Pos':<5} {'Type':<8} {'Layers':<8} {'States':<8} {'MAE':<12} {'RMSE':<12}")
    print("-"*78)

    pos_types = ['corner', 'edge', 'corner', 'edge', 'center', 'edge', 'corner', 'edge', 'corner']

    for stat in summary_stats:
        pos = stat['position']
        print(f"{pos:<5} {pos_types[pos]:<8} {stat['n_layers']:<8} {stat['n_states']:<8} "
              f"{stat['mae']:<12.4f} {stat['rmse']:<12.4f}")

    print("-"*78)
    avg_mae = np.mean([s['mae'] for s in summary_stats])
    avg_rmse = np.mean([s['rmse'] for s in summary_stats])
    print(f"{'AVG':<5} {'':<8} {'':<8} {'':<8} {avg_mae:<12.4f} {avg_rmse:<12.4f}")
    print("="*78)

    # Save text summary
    with open('fourier_branch_results.txt', 'w') as f:
        f.write("FOURIER BRANCH ANALYSIS - RESULTS SUMMARY\n")
        f.write("="*78 + "\n\n")
        f.write(f"{'Pos':<5} {'Type':<8} {'Layers':<8} {'States':<8} {'MAE':<12} {'RMSE':<12}\n")
        f.write("-"*78 + "\n")
        for stat in summary_stats:
            pos = stat['position']
            f.write(f"{pos:<5} {pos_types[pos]:<8} {stat['n_layers']:<8} {stat['n_states']:<8} "
                   f"{stat['mae']:<12.4f} {stat['rmse']:<12.4f}\n")
        f.write("-"*78 + "\n")
        f.write(f"{'AVG':<5} {'':<8} {'':<8} {'':<8} {avg_mae:<12.4f} {avg_rmse:<12.4f}\n")
        f.write("="*78 + "\n")
    print("  ✓ Saved fourier_branch_results.txt")

    print("\nAnalysis complete!")
    print(f"Generated {len(all_results)} position analyses")
    print("Check position_N_fourier_branch.png for visualizations (N=0-8)")


if __name__ == '__main__':
    main()
