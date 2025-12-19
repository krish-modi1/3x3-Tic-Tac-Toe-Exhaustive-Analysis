import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

# Set matplotlib style
plt.style.use('default')

def build_sorted_plot_data(position_df):
    """
    Build plot data with proper sorting by layer using parent_state column.
    """
    plot_data = []
    if len(position_df) == 0:
        return pd.DataFrame(columns=['layer', 'state', 'px', 'x_order', 'parent_state'])

    min_layer = position_df['layer'].min()
    max_layer = position_df['layer'].max()
    
    # Layer 1 (Root)
    layer_1 = position_df[position_df['layer'] == min_layer]
    if len(layer_1) > 0:
        row = layer_1.iloc[0]
        plot_data.append({
            'layer': min_layer,
            'state': row['state'],
            'px': row['P(X wins)'],
            'x_order': 0,
            'parent_state': row.get('parent_state', None)
        })

    # Layer 2
    if min_layer + 1 <= max_layer:
        layer_2 = position_df[position_df['layer'] == (min_layer + 1)].sort_values('P(X wins)').reset_index(drop=True)
        for idx, (_, row) in enumerate(layer_2.iterrows()):
            plot_data.append({
                'layer': min_layer + 1,
                'state': row['state'],
                'px': row['P(X wins)'],
                'x_order': idx,
                'parent_state': row['parent_state']
            })

    # Layer 3+
    for layer in range(min_layer + 2, max_layer + 1):
        layer_data = position_df[position_df['layer'] == layer].copy()
        if len(layer_data) == 0:
            continue

        parent_layer = layer - 1
        parent_plot_data = [d for d in plot_data if d['layer'] == parent_layer]
        
        # If no parents survived, skip
        if not parent_plot_data:
            continue
            
        # Sort parents by x_order
        parent_plot_data_sorted = sorted(parent_plot_data, key=lambda x: x['x_order'])
        parent_states_ordered = [d['state'] for d in parent_plot_data_sorted]
        
        # Map children
        parent_to_children = defaultdict(list)
        for _, child_row in layer_data.iterrows():
            parent_state = child_row['parent_state']
            if parent_state in parent_states_ordered:
                parent_to_children[parent_state].append({
                    'state': child_row['state'],
                    'px': child_row['P(X wins)'],
                    'parent_state': parent_state
                })

        # Assign x_order
        x_order_counter = 0
        for parent_state in parent_states_ordered:
            children = parent_to_children.get(parent_state, [])
            children.sort(key=lambda x: x['px'])
            for child in children:
                plot_data.append({
                    'layer': layer,
                    'state': child['state'],
                    'px': child['px'],
                    'x_order': x_order_counter,
                    'parent_state': child['parent_state']
                })
                x_order_counter += 1

    return pd.DataFrame(plot_data)

def generate_layer_avg_analysis(position_file):
    if not os.path.exists(position_file):
        print("File not found.")
        return

    # --- 1. Load and Prepare Data ---
    df = pd.read_csv(position_file)
    
    # Identify Fuzzy States
    df['is_determined'] = df[['P(X wins)', 'P(O wins)', 'P(draws)']].max(axis=1) > 0.9999
    
    print("Building ordering for Original Data (All States)...")
    original_plot_df = build_sorted_plot_data(df)
    
    print("Building ordering for Fuzzy Data (Methodology)...")
    fuzzy_df = df[~df['is_determined']].copy()
    fuzzy_plot_df = build_sorted_plot_data(fuzzy_df)
    
    # --- 2. Calculate Weights and Centered Probs (RECURSIVE LAYER AVERAGE) ---
    state_adj_prob_map = {}
    
    layers = sorted(fuzzy_plot_df['layer'].unique())
    min_layer = min(layers) if layers else 1
    
    # Initialize Layer 1
    l1_data = fuzzy_plot_df[fuzzy_plot_df['layer'] == min_layer]
    for _, row in l1_data.iterrows():
        state_adj_prob_map[row['state']] = row['px']
        
    fuzzy_plot_df['parent_adj_prob'] = 0.0
    fuzzy_plot_df['weight'] = 0.0
    fuzzy_plot_df['centered_prob'] = 0.0
    
    fuzzy_plot_df.loc[fuzzy_plot_df['layer'] == min_layer, 'centered_prob'] = fuzzy_plot_df.loc[fuzzy_plot_df['layer'] == min_layer, 'px']

    print("Running Recursive Layer-Average Adjustment...")
    for layer in layers:
        if layer == min_layer: continue
        
        layer_mask = fuzzy_plot_df['layer'] == layer
        layer_data = fuzzy_plot_df[layer_mask]
        
        parent_groups = layer_data.groupby('parent_state')
        
        # Pass 1: Calculate Local Weights
        local_weights = []
        
        for parent, group in parent_groups:
            p_parent_adj = state_adj_prob_map.get(parent)
            if p_parent_adj is None: p_parent_adj = 0.0
            
            mean_child_raw = group['px'].mean()
            
            if abs(p_parent_adj) > 1e-9:
                w = (mean_child_raw - 0.5) / p_parent_adj
                local_weights.append(w)
            # Else skip or weight 0
            
        # Calculate Layer Average Weight
        if local_weights:
            w_layer_avg = sum(local_weights) / len(local_weights)
        else:
            w_layer_avg = 0.0
            
        # Pass 2: Apply Average Weight
        for parent, group in parent_groups:
            p_parent_adj = state_adj_prob_map.get(parent)
            if p_parent_adj is None: p_parent_adj = 0.0
            
            # Apply the LAYER AVERAGE weight
            indices = group.index
            fuzzy_plot_df.loc[indices, 'weight'] = w_layer_avg
            fuzzy_plot_df.loc[indices, 'parent_adj_prob'] = p_parent_adj
            
            new_probs = group['px'] - (w_layer_avg * p_parent_adj)
            fuzzy_plot_df.loc[indices, 'centered_prob'] = new_probs
            
            # Update Map
            for idx, val in new_probs.items():
                state = fuzzy_plot_df.loc[idx, 'state']
                state_adj_prob_map[state] = val

    output_csv = 'position_4_layer_avg_analysis.csv'
    fuzzy_plot_df.to_csv(output_csv, index=False)
    print(f"Detailed analysis saved to {output_csv}")
    
    # Define common ylim
    common_ylim = (-0.05, 1.05)

    # --- 3. Generate PNG 1: Original Position Analysis ---
    print("Generating Original Plot...")
    fig1, axes1 = plt.subplots(3, 3, figsize=(18, 15))
    axes1 = axes1.flatten()
    
    for idx, layer in enumerate(range(1, 10)):
        ax = axes1[idx]
        layer_data = original_plot_df[original_plot_df['layer'] == layer]
        
        if len(layer_data) > 0:
            ax.plot(layer_data['x_order'], layer_data['px'], 'k-', alpha=0.3, linewidth=1.0, zorder=1)
            ax.scatter(layer_data['x_order'], layer_data['px'], c='blue', alpha=0.8, s=20, zorder=2)
            ax.set_title(f'Layer {layer} (Original Raw)', fontsize=12, fontweight='bold')
            ax.set_ylim(common_ylim) 
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx >= 6: ax.set_xlabel('Index')

    plt.suptitle('Original Position Analysis (Raw Probabilities)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('final_original_layer_avg.png', dpi=150)
    plt.close()
    
    # --- 4. Generate PNG 2: Methodology Analysis (Layer Avg) ---
    print("Generating Methodology Plot (Layer Avg)...")
    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 15))
    axes2 = axes2.flatten()
    
    for idx, layer in enumerate(range(1, 10)):
        ax = axes2[idx]
        layer_data = fuzzy_plot_df[fuzzy_plot_df['layer'] == layer]
        
        if len(layer_data) > 0:
            colors = layer_data['centered_prob'].apply(lambda x: 'green' if x >= 0.5 else 'red')
            
            ax.plot(layer_data['x_order'], layer_data['centered_prob'], 'k-', alpha=0.3, linewidth=1.0, zorder=1)
            ax.scatter(layer_data['x_order'], layer_data['centered_prob'], c=colors, alpha=0.8, s=20, zorder=2)
            ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5, zorder=3)
            
            # Add annotation for the Weight used
            avg_w = layer_data['weight'].iloc[0] # All same for layer
            ax.text(0.05, 0.95, f"W_avg: {avg_w:.3f}", transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
            
            ax.set_title(f'Layer {layer} (Layer-Avg Centered)', fontsize=12, fontweight='bold')
            ax.set_ylim(common_ylim) 
        else:
            status = "ALL SOLVED" if layer > 1 else "No Data"
            ax.text(0.5, 0.5, status, ha='center', va='center', color='black', fontsize=12)
            
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx >= 6: ax.set_xlabel('Index')

    plt.suptitle('New Methodology: Recursive + Layer Average Weight', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('final_methodology_layer_avg.png', dpi=150)
    plt.close()

# Execute
generate_layer_avg_analysis('position_4_probabilities.csv')