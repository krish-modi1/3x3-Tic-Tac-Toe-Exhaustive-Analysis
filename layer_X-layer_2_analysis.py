import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def load_and_build_genealogy(position, csv_dir='position_probabilities'):
    """
    Load probabilities and build Layer 2 branch genealogy.

    Args:
        position: Opening position index (0-8)
        csv_dir: Directory containing probability CSV files

    Returns:
        Tuple of (full_df, layer2_df, genealogy_map)
        - full_df: All states with probabilities
        - layer2_df: Layer 2 states only
        - genealogy_map: {state -> layer2_ancestor} for layers 3-8
    """
    csv_path = os.path.join(csv_dir, f'position_{position}_probabilities.csv')

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None, None, None

    df = pd.read_csv(csv_path)

    # Filter states where X is at the opening position
    position_df = df[df['state'].str[position] == 'X'].copy()

    # Extract Layer 2 states (branch roots)
    layer2_df = position_df[position_df['layer'] == 2].copy()

    # Build genealogy: map each state to its Layer 2 ancestor
    genealogy_map = {}

    # Initialize: Layer 2 states are their own ancestors
    layer2_states = set(layer2_df['state'].values)
    for state in layer2_states:
        genealogy_map[state] = state

    # Build descendants for layers 3-8
    descendants_by_ancestor = {l2_state: {l2_state} for l2_state in layer2_states}

    for layer in range(3, 9):
        layer_states = position_df[position_df['layer'] == layer]

        for _, row in layer_states.iterrows():
            state = row['state']
            parent = row['parent_state']

            # Find which Layer 2 ancestor this descends from
            for l2_ancestor, descendants in descendants_by_ancestor.items():
                if parent in descendants:
                    descendants.add(state)
                    genealogy_map[state] = l2_ancestor
                    break

    print(f"Position {position}: Loaded {len(position_df)} states")
    print(f"  Layer 2 roots: {len(layer2_df)}")
    print(f"  Descendants (L3-8): {len(genealogy_map) - len(layer2_states)}")

    return position_df, layer2_df, genealogy_map

def compute_layer2_reference_differences(position_df, layer2_df, genealogy_map):
    """
    Compute probability differences relative to Layer 2 ancestors.

    Args:
        position_df: DataFrame with all states
        layer2_df: DataFrame with Layer 2 states
        genealogy_map: Mapping from state to Layer 2 ancestor

    Returns:
        DataFrame with Layer 2 reference differences
    """
    # Build Layer 2 probability lookup
    layer2_probs = {}
    for _, row in layer2_df.iterrows():
        layer2_probs[row['state']] = {
            'px': row['P(X wins)'],
            'po': row['P(O wins)'],
            'pd': row['P(draws)']
        }

    # Compute differences for layers 3-8
    diff_data = []

    for _, row in position_df.iterrows():
        state = row['state']
        layer = row['layer']

        # Skip layers 0-2 (no differences computed)
        if layer < 3:
            continue

        # Get Layer 2 ancestor
        if state not in genealogy_map:
            continue

        l2_ancestor = genealogy_map[state]
        l2_probs = layer2_probs[l2_ancestor]

        # Compute differences from Layer 2 ancestor
        px_diff = row['P(X wins)'] - l2_probs['px']
        po_diff = row['P(O wins)'] - l2_probs['po']
        pd_diff = row['P(draws)'] - l2_probs['pd']

        diff_data.append({
            'state': state,
            'layer2_ancestor': l2_ancestor,
            'immediate_parent': row['parent_state'],
            'to_move': row['to_move'],
            'layer': layer,
            'P(X wins)': row['P(X wins)'],
            'P(X wins) from L2': px_diff,
            'P(O wins)': row['P(O wins)'],
            'P(O wins) from L2': po_diff,
            'P(draws)': row['P(draws)'],
            'P(draws) from L2': pd_diff
        })

    return pd.DataFrame(diff_data)

def build_sorted_plot_data(position_df, layer2_df, diff_df, prob_type='P(X wins)'):
    """
    Build plot data with proper sorting by layer.

    Sorting methodology:
    - Layers 0-2: Use absolute probabilities (no differences)
    - Layers 3-8: Group by layer2_ancestor, sort within groups by difference ascending

    Args:
        position_df: Full DataFrame with all states
        layer2_df: Layer 2 states DataFrame
        diff_df: Difference DataFrame for layers 3-8
        prob_type: Which probability to analyze

    Returns:
        DataFrame with x_order column for plotting
    """
    plot_data = []

    # Layer 1: Single opening state
    layer_1 = position_df[position_df['layer'] == 1]
    if len(layer_1) > 0:
        plot_data.append({
            'layer': 1,
            'state': layer_1.iloc[0]['state'],
            'value': layer_1.iloc[0][prob_type],
            'is_difference': False,
            'x_order': 0
        })

    # Layer 2: Sort by absolute probability ascending
    layer_2 = layer2_df.sort_values(prob_type).reset_index(drop=True)
    for idx, (_, row) in enumerate(layer_2.iterrows()):
        plot_data.append({
            'layer': 2,
            'state': row['state'],
            'value': row[prob_type],
            'is_difference': False,
            'x_order': idx
        })

    # Layers 3-8: Group by layer2_ancestor, sort by difference
    diff_col = f'{prob_type} from L2'

    for layer in range(3, 9):
        layer_data = diff_df[diff_df['layer'] == layer].copy()
        if len(layer_data) == 0:
            break

        # Get Layer 2 ancestor order from layer 2 plot data
        layer2_order = [d['state'] for d in plot_data if d['layer'] == 2]

        # Group by layer2_ancestor
        ancestor_to_children = defaultdict(list)

        for _, child_row in layer_data.iterrows():
            l2_ancestor = child_row['layer2_ancestor']

            if l2_ancestor in layer2_order:
                ancestor_to_children[l2_ancestor].append({
                    'state': child_row['state'],
                    'value': child_row[diff_col]
                })

        # Build x_order: iterate through Layer 2 ancestors in order
        x_order_counter = 0
        for l2_ancestor in layer2_order:
            children = ancestor_to_children.get(l2_ancestor, [])

            # Sort children by difference ascending
            children.sort(key=lambda x: x['value'])

            for child in children:
                plot_data.append({
                    'layer': layer,
                    'state': child['state'],
                    'value': child['value'],
                    'is_difference': True,
                    'x_order': x_order_counter
                })
                x_order_counter += 1

    return pd.DataFrame(plot_data)

def create_layer2_reference_plots(plot_df, position, prob_type, output_file):
    """
    Create 3x3 grid of plots showing Layer 2 reference analysis.

    Layers 0-2: Absolute probabilities (no differences)
    Layers 3-8: Differences from Layer 2 ancestor

    Args:
        plot_df: DataFrame with plot data
        position: Opening position index
        prob_type: Which probability type
        output_file: Output filename
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    title_map = {
        'P(X wins)': 'P(X wins) from Layer 2',
        'P(O wins)': 'P(O wins) from Layer 2',
        'P(draws)': 'P(draws) from Layer 2'
    }

    fig.suptitle(f'Position {position} Opening: {title_map[prob_type]}', 
                 fontsize=14, fontweight='bold')

    axes_flat = axes.flatten()

    for plot_idx in range(9):
        ax = axes_flat[plot_idx]
        layer = plot_idx

        layer_plot_data = plot_df[plot_df['layer'] == layer].sort_values('x_order')

        if len(layer_plot_data) == 0:
            ax.text(0.5, 0.5, f'Layer {layer}: No data', ha='center', va='center')
            ax.set_title(f'Layer {layer}')
            continue

        x = layer_plot_data['x_order'].values
        y = layer_plot_data['value'].values
        is_diff = layer_plot_data['is_difference'].iloc[0] if len(layer_plot_data) > 0 else False

        # Plot styling
        ax.plot(x, y, 'o-', color='steelblue', linewidth=1.5, markersize=4)
        ax.fill_between(x, y, alpha=0.2, color='steelblue')

        # Add horizontal reference line for difference plots
        if is_diff:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('State Index', fontsize=9)

        if is_diff:
            ax.set_ylabel('Difference from Layer 2', fontsize=9)
            title_suffix = '(Δ from L2)'
        else:
            ax.set_ylabel('Probability', fontsize=9)
            ax.set_ylim([-0.05, 1.05])
            title_suffix = '(Absolute)'

        ax.set_title(f'Layer {layer} ({len(layer_plot_data)} states) {title_suffix}', 
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        min_v = y.min()
        max_v = y.max()
        mean_v = y.mean()

        stats_text = f'Min: {min_v:.3f}\nMax: {max_v:.3f}\nMean: {mean_v:.3f}'
        ax.text(0.98, 0.02, stats_text, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                transform=ax.transAxes, family='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == '__main__':
    n = 3
    csv_dir = 'position_probabilities'

    # Create output directories
    output_dirs = {
        'P(X wins)': 'layer2_reference_plots/x_wins',
        'P(O wins)': 'layer2_reference_plots/o_wins',
        'P(draws)': 'layer2_reference_plots/draws'
    }

    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)

    print(f"Generating Layer 2 reference analysis for all {n}x{n} opening positions")
    print("="*70)

    for position in range(n * n):
        print(f"\nAnalyzing position {position}...")

        # Load data and build genealogy
        position_df, layer2_df, genealogy_map = load_and_build_genealogy(position, csv_dir)

        if position_df is None or layer2_df is None:
            print(f"  Skipping position {position} (no data)")
            continue

        # Compute differences from Layer 2 ancestors
        diff_df = compute_layer2_reference_differences(position_df, layer2_df, genealogy_map)
        print(f"  Computed {len(diff_df)} Layer 2 reference differences")

        # Generate plots for each probability type
        for prob_type, output_dir in output_dirs.items():
            print(f"  Generating {prob_type} plot...")

            plot_df = build_sorted_plot_data(position_df, layer2_df, diff_df, prob_type)

            print(f"    Plot data summary:")
            for layer in sorted(plot_df['layer'].unique()):
                count = len(plot_df[plot_df['layer'] == layer])
                is_diff = plot_df[plot_df['layer'] == layer]['is_difference'].iloc[0] if count > 0 else False
                diff_str = 'differences' if is_diff else 'absolute values'
                print(f"      Layer {layer}: {count} states ({diff_str})")

            output_file = os.path.join(output_dir, f'position_{position}_l2ref_analysis.png')
            create_layer2_reference_plots(plot_df, position, prob_type, output_file)

    print("\n" + "="*70)
    print(f"Complete! Generated {n * n * 3} plot files:")
    for prob_type, output_dir in output_dirs.items():
        print(f"  {prob_type}: {output_dir}/")
