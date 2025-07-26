import numpy as np
from hmmlearn import hmm

move_map = {'R': 0, 'P': 1, 'S': 2}
reverse_move_map = {v: k for k, v in move_map.items()}

def counter_move(move_num):
    return (move_num + 1) % 3

class RPS_HMM_Player:
    def __init__(self):
        # Freeze all parameters so they are NOT re-initialized/refit
        self.model = hmm.MultinomialHMM(n_components=3, n_iter=10, tol=0.01, init_params="", params="")
        self.model.n_trials = 1  # One trial per sample

        self.model.startprob_ = np.array([1/3, 1/3, 1/3])
        self.model.transmat_ = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        self.model.emissionprob_ = np.array([
            [0.7, 0.15, 0.15],
            [0.15, 0.7, 0.15],
            [0.15, 0.15, 0.7],
        ])

        self.moves_observed = []

    def one_hot_encode(self, moves):
        n_samples = len(moves)
        one_hot = np.zeros((n_samples, 3), dtype=int)
        for i, move in enumerate(moves):
            one_hot[i, move] = 1
        return one_hot

    def update_model(self):
        if len(self.moves_observed) < 5:
            return
        X = self.one_hot_encode(self.moves_observed)
        # Set n_trials to sum of each row (which is 1 for one-hot)
        self.model.n_trials = 1
        # Because params="" we must update model parameters manually or do not call fit here.
        # Instead, consider skipping fit or carefully implementing incremental learning.
        # If you want to fit, you must allow init_params and params to include emission and transmat.
        # For now, just skip fitting to avoid parameter overwrites.
        pass

    def predict_next_move(self):
        if len(self.moves_observed) < 2:
            return np.random.choice([0, 1, 2])
        X = self.one_hot_encode(self.moves_observed)
        self.model.n_trials = 1
        # Do NOT call fit here or it will overwrite parameters and cause instability
        # Instead, decode using current model parameters
        logprob, hidden_states = self.model.decode(X, algorithm="viterbi")
        last_state = hidden_states[-1]
        next_state_prob = self.model.transmat_[last_state]
        expected_move_prob = np.dot(next_state_prob, self.model.emissionprob_)
        predicted_move = np.argmax(expected_move_prob)
        return predicted_move

    def record_move(self, move_char):
        if move_char in move_map:
            self.moves_observed.append(move_map[move_char])

def main():
    player = RPS_HMM_Player()

    print("Let's play Rock-Paper-Scissors!")
    print("Enter your move: R (rock), P (paper), S (scissors), or Q to quit.")

    while True:
        user_move = input("Your move: ").strip().upper()
        if user_move == 'Q':
            print("Thanks for playing!")
            break
        if user_move not in move_map:
            print("Invalid move. Please enter R, P, S, or Q.")
            continue

        player.record_move(user_move)
        player.update_model()

        predicted_move = player.predict_next_move()
        bot_move_num = counter_move(predicted_move)
        bot_move = reverse_move_map[bot_move_num]

        print(f"Bot plays: {bot_move}")

        user_num = move_map[user_move]
        if user_num == bot_move_num:
            print("Bot wins! 🎉")
        elif user_num == (bot_move_num + 1) % 3:
            print("You win! 🏆")
        else:
            print("It's a tie! 🤝")

if __name__ == "__main__":
    main()
