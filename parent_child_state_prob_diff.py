import pandas as pd
import os

def compute_probability_differences(position, csv_dir='position_probabilities', output_dir='position_probability_differences'):
    """
    Compute probability differences between parent and child states.

    For each state, calculates:
    - P(X wins) difference = child P(X wins) - parent P(X wins)
    - P(O wins) difference = child P(O wins) - parent P(O wins)
    - P(draws) difference = child P(draws) - parent P(draws)

    Args:
        position: Opening position index (0-8)
        csv_dir: Directory containing probability CSV files
        output_dir: Directory to save difference CSV files

    Returns:
        DataFrame with probability differences
    """
    csv_path = os.path.join(csv_dir, f'position_{position}_probabilities.csv')

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    # Filter states where X is at the opening position
    position_df = df[df['state'].str[position] == 'X'].copy()

    # Create state to probability lookup dictionary
    state_to_probs = {}
    for _, row in position_df.iterrows():
        state_to_probs[row['state']] = {
            'px': row['P(X wins)'],
            'po': row['P(O wins)'],
            'pd': row['P(draws)']
        }

    # Compute differences for each state with a parent
    diff_data = []

    for _, row in position_df.iterrows():
        state = row['state']
        parent_state = row['parent_state']

        # Skip if no parent (root state at layer 1)
        if pd.isna(parent_state) or parent_state == '':
            continue

        # Get parent probabilities
        if parent_state in state_to_probs:
            parent_probs = state_to_probs[parent_state]
            child_probs = state_to_probs[state]

            # Compute differences
            px_diff = child_probs['px'] - parent_probs['px']
            po_diff = child_probs['po'] - parent_probs['po']
            pd_diff = child_probs['pd'] - parent_probs['pd']

            diff_data.append({
                'state': state,
                'parent_state': parent_state,
                'to_move': row['to_move'],
                'layer': row['layer'],
                'P(X wins)': child_probs['px'],
                'P(X wins) difference': px_diff,
                'P(O wins)': child_probs['po'],
                'P(O wins) difference': po_diff,
                'P(draws)': child_probs['pd'],
                'P(draws) difference': pd_diff
            })

    diff_df = pd.DataFrame(diff_data)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'position_{position}_probability_differences.csv')
    diff_df.to_csv(output_file, index=False)

    print(f"Position {position}: Computed {len(diff_df)} probability differences")
    print(f"  Saved to {output_file}")

    return diff_df

if __name__ == '__main__':
    n = 3
    csv_dir = 'position_probabilities'
    output_dir = 'position_probability_differences'

    print(f"Computing probability differences for all {n}x{n} opening positions")
    print("="*70)

    for position in range(n * n):
        print(f"\nProcessing position {position}...")

        diff_df = compute_probability_differences(position, csv_dir, output_dir)

        if diff_df is not None and len(diff_df) > 0:
            print(f"  Layer distribution:")
            for layer in sorted(diff_df['layer'].unique()):
                count = len(diff_df[diff_df['layer'] == layer])
                print(f"    Layer {layer}: {count} states")

    print("\n" + "="*70)
    print(f"Complete! Generated {n * n} difference CSV files in '{output_dir}/' directory")
