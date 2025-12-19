import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def load_and_prepare_data(position, csv_dir='position_probabilities'):
    """
    Load exhaustive probabilities for a specific opening position.

    Args:
        position: Opening position index (0-8)
        csv_dir: Directory containing probability CSV files

    Returns:
        DataFrame with probability data for the specified position
    """
    csv_path = os.path.join(csv_dir, f'position_{position}_probabilities.csv')

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    # Filter states where X is at the opening position
    position_df = df[df['state'].str[position] == 'X'].copy().reset_index(drop=True)

    print(f"Position {position}: Loaded {len(position_df)} states")

    return position_df

def build_sorted_plot_data(position_df):
    """
    Build plot data with proper sorting by layer using parent_state column.

    Sorting methodology:
    - Layer 1: Single opening state
    - Layer 2: All states sorted by P(X wins) ascending
    - Layer 3+: States grouped by parent, within-group sorted by P(X wins) ascending

    The parent_state column allows direct lookup instead of testing all possible parents.

    Args:
        position_df: DataFrame containing states for one opening position

    Returns:
        DataFrame with x_order column for plotting
    """
    plot_data = []

    # Layer 1: Opening move
    layer_1 = position_df[position_df['layer'] == 1]
    if len(layer_1) > 0:
        plot_data.append({
            'layer': 1,
            'state': layer_1.iloc[0]['state'],
            'px': layer_1.iloc[0]['P(X wins)'],
            'x_order': 0
        })

    # Layer 2: Sort all states by P(X wins) ascending
    layer_2 = position_df[position_df['layer'] == 2].sort_values('P(X wins)').reset_index(drop=True)
    for idx, (_, row) in enumerate(layer_2.iterrows()):
        plot_data.append({
            'layer': 2,
            'state': row['state'],
            'px': row['P(X wins)'],
            'x_order': idx
        })

    # Layer 3+: Group by parent using parent_state column
    for layer in range(3, 10):
        layer_data = position_df[position_df['layer'] == layer].copy()
        if len(layer_data) == 0:
            break

        parent_layer = layer - 1

        # Get ordered parent states from previous layer
        parent_plot_data = [d for d in plot_data if d['layer'] == parent_layer]
        parent_states_ordered = [d['state'] for d in sorted(parent_plot_data, key=lambda x: x['x_order'])]

        # Group children by parent_state column (direct lookup, no testing needed)
        parent_to_children = defaultdict(list)

        for _, child_row in layer_data.iterrows():
            parent_state = child_row['parent_state']

            # Only process if parent is in our ordered list
            if parent_state in parent_states_ordered:
                parent_to_children[parent_state].append({
                    'state': child_row['state'],
                    'px': child_row['P(X wins)']
                })

        # Build x_order: iterate through parents in order, sort children within each group
        x_order_counter = 0
        for parent_state in parent_states_ordered:
            children = parent_to_children.get(parent_state, [])

            # Sort children by P(X wins) ascending
            children.sort(key=lambda x: x['px'])

            for child in children:
                plot_data.append({
                    'layer': layer,
                    'state': child['state'],
                    'px': child['px'],
                    'x_order': x_order_counter
                })
                x_order_counter += 1

    return pd.DataFrame(plot_data)

def create_line_plots(plot_df, position, output_file):
    """
    Create 3x3 grid of line plots for one opening position.

    Args:
        plot_df: DataFrame with plot data (layer, state, px, x_order)
        position: Opening position index (0-8)
        output_file: Output filename for the plot
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'Position {position} Opening: P(X wins) by Layer', fontsize=14, fontweight='bold')

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
        y = layer_plot_data['px'].values

        ax.plot(x, y, 'o-', color='steelblue', linewidth=1.5, markersize=4)
        ax.fill_between(x, y, alpha=0.2, color='steelblue')

        ax.set_xlabel('State Index', fontsize=9)
        ax.set_ylabel('P(X wins)', fontsize=9)
        ax.set_title(f'Layer {layer} ({len(layer_plot_data)} states)', fontsize=11, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        min_p, max_p, mean_p = y.min(), y.max(), y.mean()
        stats_text = f'Min: {min_p:.3f}\nMax: {max_p:.3f}\nMean: {mean_p:.3f}'
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
    output_dir = 'position_plots'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating analysis plots for all {n}x{n} opening positions")
    print("="*70)

    for position in range(n * n):
        print(f"\nAnalyzing position {position}...")

        position_df = load_and_prepare_data(position, csv_dir)

        if position_df is None or len(position_df) == 0:
            print(f"  Skipping position {position} (no data)")
            continue

        plot_df = build_sorted_plot_data(position_df)

        print(f"  Plot data summary:")
        for layer in sorted(plot_df['layer'].unique()):
            count = len(plot_df[plot_df['layer'] == layer])
            print(f"    Layer {layer}: {count} states")

        output_file = os.path.join(output_dir, f'position_{position}_analysis.png')
        create_line_plots(plot_df, position, output_file)

    print("\n" + "="*70)
    print(f"Complete! Generated {n * n} plot files in '{output_dir}/' directory")
