import json
import csv
import os
from collections import defaultdict

class TicTacToeProbabilityCalculator:
    def __init__(self, n: int):
        self.n = n
        self.board_size = n * n
        self.game_tree = {}
        self.parent_map = {}

    def load_tree_from_json(self, filename):
        """
        Load game tree from JSON file and build parent-child relationships.

        Args:
            filename: Path to JSON file containing game tree

        Returns:
            Dictionary containing loaded game tree statistics
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        tree = data['tree']
        game_tree = {}
        parent_map = {}

        for key_str, node_data in tree.items():
            key_list = eval(key_str)
            state = tuple(key_list)

            moves = []
            if not node_data['terminal']:
                for mv, child_list in node_data['moves']:
                    child_state = tuple(child_list)
                    moves.append((mv, child_state))

                    # Record parent-child relationship
                    parent_map[child_state] = state

            game_tree[state] = {
                'moves': moves,
                'terminal': node_data['terminal'],
                'result': node_data['result'],
                'player': node_data['player']
            }

        self.game_tree = game_tree
        self.parent_map = parent_map
        return data['statistics']

    def compute_exhaustive_probabilities(self):
        """
        Compute P(X wins), P(O wins), P(draw) for each state under exhaustive play.
        For each state, uniformly explore all possible continuations.

        Returns:
            Dictionary mapping state tuples to (P(X wins), P(O wins), P(draw))
        """
        probabilities = {}

        def compute_probs(state):
            if state in probabilities:
                return probabilities[state]

            node = self.game_tree[state]

            if node['terminal']:
                result = node['result']
                if result == 1:
                    probs = (1.0, 0.0, 0.0)
                elif result == -1:
                    probs = (0.0, 1.0, 0.0)
                else:
                    probs = (0.0, 0.0, 1.0)
            else:
                moves = node['moves']
                n_moves = len(moves)

                px_total = 0.0
                po_total = 0.0
                pd_total = 0.0

                for mv, child_state in moves:
                    px, po, pd = compute_probs(child_state)
                    px_total += px
                    po_total += po
                    pd_total += pd

                probs = (px_total / n_moves, po_total / n_moves, pd_total / n_moves)

            probabilities[state] = probs
            return probs

        for state in self.game_tree:
            compute_probs(state)

        return probabilities

    def save_probabilities_to_csv(self, probabilities, filename):
        """
        Save state probabilities to CSV with parent_state column.

        Args:
            probabilities: Dictionary of state -> (px, po, pd)
            filename: Output CSV filename
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['state', 'parent_state', 'to_move', 'layer', 'P(X wins)', 'P(O wins)', 'P(draws)'])

            for state, probs in probabilities.items():
                board = list(state)
                node = self.game_tree[state]

                # Convert state to string representation
                state_str = ''.join('.' if x == 0 else ('X' if x == 1 else 'O') for x in board)

                # Get parent state string representation
                parent_state_str = ''
                if state in self.parent_map:
                    parent_board = list(self.parent_map[state])
                    parent_state_str = ''.join('.' if x == 0 else ('X' if x == 1 else 'O') for x in parent_board)

                player = node['player']
                to_move = 'X' if player == 1 else 'O'

                layer = sum(1 for x in board if x != 0)

                px, po, pd = probs

                writer.writerow([state_str, parent_state_str, to_move, layer, 
                               f'{px:.10f}', f'{po:.10f}', f'{pd:.10f}'])

        print(f"Probabilities saved to {filename}")

if __name__ == '__main__':
    n = 3

    input_dir = 'game_tree'
    output_dir = 'position_probabilities'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Computing exhaustive probabilities for all {n}x{n} opening positions")
    print("="*70)

    calculator = TicTacToeProbabilityCalculator(n)

    for position in range(n * n):
        print(f"\nProcessing position {position}...")

        input_file = os.path.join(input_dir, f'position_{position}.json')

        if not os.path.exists(input_file):
            print(f"  Warning: {input_file} not found, skipping")
            continue

        stats = calculator.load_tree_from_json(input_file)
        print(f"  Loaded {stats['total_states']} states")

        probabilities = calculator.compute_exhaustive_probabilities()
        print(f"  Computed probabilities for {len(probabilities)} states")

        output_file = os.path.join(output_dir, f'position_{position}_probabilities.csv')
        calculator.save_probabilities_to_csv(probabilities, output_file)

    print("\n" + "="*70)
    print(f"Complete! Generated {n * n} CSV files in '{output_dir}/' directory")
