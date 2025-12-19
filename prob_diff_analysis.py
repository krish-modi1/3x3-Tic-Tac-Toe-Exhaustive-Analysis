import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def load_difference_data(position, csv_dir='position_probability_differences'):
    """
    Load probability difference data for a specific opening position.

    Args:
        position: Opening position index (0-8)
        csv_dir: Directory containing probability difference CSV files

    Returns:
        DataFrame with probability difference data
    """
    csv_path = os.path.join(csv_dir, f'position_{position}_probability_differences.csv')

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    print(f"Position {position}: Loaded {len(df)} state differences")

    return df

def build_sorted_plot_data(diff_df, prob_type='P(X wins)'):
    """
    Build plot data with proper sorting by layer using parent_state column.

    Sorting methodology:
    - Layer 2: All states sorted by probability difference ascending
    - Layer 3+: States grouped by parent, within-group sorted by probability difference ascending

    Args:
        diff_df: DataFrame containing probability differences
        prob_type: Which probability to analyze ('P(X wins)', 'P(O wins)', or 'P(draws)')

    Returns:
        DataFrame with x_order column for plotting
    """
    plot_data = []

    diff_col = f'{prob_type} difference'

    # Layer 2: Sort all states by probability difference ascending
    layer_2 = diff_df[diff_df['layer'] == 2].sort_values(diff_col).reset_index(drop=True)
    for idx, (_, row) in enumerate(layer_2.iterrows()):
        plot_data.append({
            'layer': 2,
            'state': row['state'],
            'prob_diff': row[diff_col],
            'x_order': idx
        })

    # Layer 3+: Group by parent using parent_state column
    for layer in range(3, 10):
        layer_data = diff_df[diff_df['layer'] == layer].copy()
        if len(layer_data) == 0:
            break

        parent_layer = layer - 1

        # Get ordered parent states from previous layer
        parent_plot_data = [d for d in plot_data if d['layer'] == parent_layer]
        parent_states_ordered = [d['state'] for d in sorted(parent_plot_data, key=lambda x: x['x_order'])]

        # Group children by parent_state column
        parent_to_children = defaultdict(list)

        for _, child_row in layer_data.iterrows():
            parent_state = child_row['parent_state']

            if parent_state in parent_states_ordered:
                parent_to_children[parent_state].append({
                    'state': child_row['state'],
                    'prob_diff': child_row[diff_col]
                })

        # Build x_order: iterate through parents in order, sort children within each group
        x_order_counter = 0
        for parent_state in parent_states_ordered:
            children = parent_to_children.get(parent_state, [])

            # Sort children by probability difference ascending
            children.sort(key=lambda x: x['prob_diff'])

            for child in children:
                plot_data.append({
                    'layer': layer,
                    'state': child['state'],
                    'prob_diff': child['prob_diff'],
                    'x_order': x_order_counter
                })
                x_order_counter += 1

    return pd.DataFrame(plot_data)

def create_difference_plots(plot_df, position, prob_type, output_file):
    """
    Create 3x3 grid of line plots showing probability differences.

    Args:
        plot_df: DataFrame with plot data (layer, state, prob_diff, x_order)
        position: Opening position index (0-8)
        prob_type: Which probability type ('P(X wins)', 'P(O wins)', or 'P(draws)')
        output_file: Output filename for the plot
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    title_map = {
        'P(X wins)': 'P(X wins) Differences',
        'P(O wins)': 'P(O wins) Differences',
        'P(draws)': 'P(draws) Differences'
    }

    fig.suptitle(f'Position {position} Opening: {title_map[prob_type]} by Layer', 
                 fontsize=14, fontweight='bold')

    axes_flat = axes.flatten()

    for plot_idx in range(9):
        ax = axes_flat[plot_idx]
        layer = plot_idx

        # Skip layer 0 and 1 (no differences, as they have no parents)
        if layer < 2:
            ax.text(0.5, 0.5, f'Layer {layer}: No parent', ha='center', va='center')
            ax.set_title(f'Layer {layer}')
            continue

        layer_plot_data = plot_df[plot_df['layer'] == layer].sort_values('x_order')

        if len(layer_plot_data) == 0:
            ax.text(0.5, 0.5, f'Layer {layer}: No data', ha='center', va='center')
            ax.set_title(f'Layer {layer}')
            continue

        x = layer_plot_data['x_order'].values
        y = layer_plot_data['prob_diff'].values

        # Use different colors for positive and negative differences
        colors = ['green' if val >= 0 else 'red' for val in y]

        ax.plot(x, y, 'o-', color='steelblue', linewidth=1.5, markersize=4)
        ax.fill_between(x, y, alpha=0.2, color='steelblue')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('State Index', fontsize=9)
        ax.set_ylabel('Probability Difference', fontsize=9)
        ax.set_title(f'Layer {layer} ({len(layer_plot_data)} states)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        min_d = y.min()
        max_d = y.max()
        mean_d = y.mean()

        stats_text = f'Min: {min_d:.3f}\nMax: {max_d:.3f}\nMean: {mean_d:.3f}'
        ax.text(0.98, 0.02, stats_text, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                transform=ax.transAxes, family='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == '__main__':
    n = 3
    csv_dir = 'position_probability_differences'

    # Create output directories for each probability type
    output_dirs = {
        'P(X wins)': 'position_diff_plots/x_wins',
        'P(O wins)': 'position_diff_plots/o_wins',
        'P(draws)': 'position_diff_plots/draws'
    }

    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)

    print(f"Generating probability difference plots for all {n}x{n} opening positions")
    print("="*70)

    for position in range(n * n):
        print(f"\nAnalyzing position {position}...")

        diff_df = load_difference_data(position, csv_dir)

        if diff_df is None or len(diff_df) == 0:
            print(f"  Skipping position {position} (no data)")
            continue

        # Generate plots for each probability type
        for prob_type, output_dir in output_dirs.items():
            print(f"  Generating {prob_type} difference plot...")

            plot_df = build_sorted_plot_data(diff_df, prob_type)

            print(f"    Plot data summary:")
            for layer in sorted(plot_df['layer'].unique()):
                count = len(plot_df[plot_df['layer'] == layer])
                print(f"      Layer {layer}: {count} states")

            output_file = os.path.join(output_dir, f'position_{position}_diff_analysis.png')
            create_difference_plots(plot_df, position, prob_type, output_file)

    print("\n" + "="*70)
    print(f"Complete! Generated {n * n * 3} plot files:")
    for prob_type, output_dir in output_dirs.items():
        print(f"  {prob_type}: {output_dir}/")
