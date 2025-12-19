from collections import defaultdict
import json
import os

class TicTacToeGameTree:
    def __init__(self, n: int):
        self.n = n
        self.board_size = n * n
        self.game_tree = {}
        self.state_count = defaultdict(int)
        self.terminal_states = defaultdict(int)

    @staticmethod
    def board_to_tuple(board):
        return tuple(board)

    @staticmethod
    def tuple_to_board(board_tuple):
        return list(board_tuple)

    def check_winner(self, board):
        n = self.n
        for i in range(n):
            s = sum(board[i*n + j] for j in range(n))
            if s == n: return 1
            if s == -n: return -1
        for j in range(n):
            s = sum(board[i*n + j] for i in range(n))
            if s == n: return 1
            if s == -n: return -1
        s = sum(board[i*n + i] for i in range(n))
        if s == n: return 1
        if s == -n: return -1
        s = sum(board[i*n + (n - 1 - i)] for i in range(n))
        if s == n: return 1
        if s == -n: return -1
        return 0

    def is_terminal(self, board):
        w = self.check_winner(board)
        if w != 0:
            return True, w
        if 0 not in board:
            return True, 0
        return False, None

    def get_valid_moves(self, board):
        return [i for i, v in enumerate(board) if v == 0]

    def make_move(self, board, pos, player):
        b = list(board)
        b[pos] = player
        return b

    def generate_subtree(self, opening_position):
        """
        Generate game tree for a specific opening position.

        Args:
            opening_position: Position (0-8) where X makes first move

        Returns:
            Dictionary containing the subtree rooted at this opening
        """
        start_board = [0] * self.board_size
        start_board[opening_position] = 1
        start_state = self.board_to_tuple(start_board)

        stack = [(start_state, -1)]
        visited = set()
        subtree = {}
        subtree_state_count = defaultdict(int)
        subtree_terminal_states = defaultdict(int)

        while stack:
            state, player = stack.pop()
            if state in visited:
                continue
            visited.add(state)

            board = self.tuple_to_board(state)
            terminal, result = self.is_terminal(board)

            if terminal:
                subtree[state] = {
                    'moves': [],
                    'terminal': True,
                    'result': result,
                    'player': player
                }
                if result == 1:
                    subtree_terminal_states['X_wins'] += 1
                elif result == -1:
                    subtree_terminal_states['O_wins'] += 1
                else:
                    subtree_terminal_states['draws'] += 1
            else:
                children = []
                for mv in self.get_valid_moves(board):
                    child_board = self.make_move(board, mv, player)
                    child_state = self.board_to_tuple(child_board)
                    children.append((mv, child_state))
                    stack.append((child_state, -player))

                subtree[state] = {
                    'moves': children,
                    'terminal': False,
                    'result': None,
                    'player': player
                }

            ply = sum(1 for x in board if x != 0)
            subtree_state_count[ply] += 1

        return subtree, subtree_state_count, subtree_terminal_states

    def save_subtree_to_file(self, opening_position, subtree, state_count, terminal_states, filename):
        """
        Save subtree to JSON file.

        Args:
            opening_position: Opening position index (0-8)
            subtree: Game tree dictionary for this opening
            state_count: State counts by layer
            terminal_states: Terminal state outcome counts
            filename: Output JSON filename
        """
        json_tree = {}
        for state_tuple, data in subtree.items():
            key = list(state_tuple)
            node = {
                'terminal': data['terminal'],
                'result': data['result'],
                'player': data['player'],
            }

            if data['terminal']:
                node['moves'] = []
            else:
                node['moves'] = [(mv, list(nxt)) for mv, nxt in data['moves']]

            json_tree[str(key)] = node

        payload = {
            'n': self.n,
            'opening_position': opening_position,
            'statistics': {
                'total_states': len(subtree),
                'states_by_moves': dict(state_count),
                'terminal_outcomes': dict(terminal_states)
            },
            'tree': json_tree
        }

        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)

        print(f"Position {opening_position} tree saved to {filename}")
        print(f"  States: {len(subtree)}, X wins: {terminal_states['X_wins']}, "
              f"O wins: {terminal_states['O_wins']}, Draws: {terminal_states['draws']}")

if __name__ == '__main__':
    n = 3

    output_dir = 'game_tree'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {n}x{n} Tic-Tac-Toe game trees for all opening positions")
    print("="*70)

    game = TicTacToeGameTree(n)

    for position in range(n * n):
        print(f"\nGenerating tree for opening position {position}...")
        subtree, state_count, terminal_states = game.generate_subtree(position)

        output_file = os.path.join(output_dir, f'position_{position}.json')
        game.save_subtree_to_file(position, subtree, state_count, terminal_states, output_file)

    print("\n" + "="*70)
    print(f"Complete! Generated {n * n} JSON files in '{output_dir}/' directory")
