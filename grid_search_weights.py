import itertools
import random
import json
import argparse
from collections import defaultdict, Counter
from hangman_v4 import HangmanSolver, update_pattern

def evaluate_solver(solver, words, weights):
    """Run solver in auto mode with given weights, return average success rate."""
    α, β, γ, δ, ε, ζ = weights
    wins = 0
    total_guesses_used = 0

    # Override solver weights dynamically
    def eig_letter_override(self, pattern, candidates, guessed):
        blanks_idx = [i for i, ch in enumerate(pattern) if ch == "_"]
        if not blanks_idx or not candidates:
            return None

        remaining_letters = set(ch for w in candidates for ch in w) - guessed
        if not remaining_letters:
            remaining_letters = set(self.letters) - guessed

        best_letter, best_score = None, -1.0
        total = len(candidates)

        for l in remaining_letters:
            # --- EIG ---
            buckets = Counter()
            for w in candidates:
                mask_bits = ["1" if w[i] == l else "0" for i in blanks_idx]
                buckets["".join(mask_bits)] += 1
            expected_remaining = sum(sz * sz for sz in buckets.values()) / total
            eig_score = 1.0 - (expected_remaining / total)  # normalize so bigger = better

            # --- Priors ---
            lp = self.letter_prior.get(l, 0.0)

            pos_score = 0.0
            for i in blanks_idx:
                pos_score += self.pos_prior[len(pattern)][i].get(l, 0.0)

            left_bigram_score = 0.0
            right_bigram_score = 0.0
            for i in blanks_idx:
                left = pattern[i-1] if i > 0 and pattern[i-1] != "_" else "^"
                right = pattern[i+1] if i < len(pattern)-1 and pattern[i+1] != "_" else "$"
                left_bigram_score += self.left_bigram[left].get(l, 0.0)
                right_bigram_score += self.right_bigram[l].get(right, 0.0)

            affix_score = self._affix_bonus(pattern, l)

            # --- Combined ---
            score = (
                α*eig_score +
                β*lp +
                γ*pos_score +
                δ*left_bigram_score +
                ε*right_bigram_score +
                ζ*affix_score
            )

            if score > best_score:
                best_score, best_letter = score, l
        return best_letter

    # Monkey patch the method with weights
    solver._eig_letter_for_word = eig_letter_override.__get__(solver, HangmanSolver)

    for hidden_word in words:
        pattern = "_" * len(hidden_word)
        guessedLetters = []
        guessesRemaining = 6

        while "_" in pattern and guessesRemaining > 0:
            output = solver.get_next_guess(pattern, guessedLetters, guessesRemaining)
            guess = output["nextGuess"]
            if not guess:
                break

            guessedLetters.append(guess)
            if guess in hidden_word:
                pattern = update_pattern(hidden_word, pattern, guess)
            else:
                guessesRemaining -= 1

        if "_" not in pattern:
            wins += 1
        total_guesses_used += (6 - guessesRemaining)

    return wins / len(words), total_guesses_used / len(words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", "-d", type=str, required=True, help="Airline dictionary path")
    args = parser.parse_args()

    solver = HangmanSolver(airline_dict_path=args.dict)

    # Load airline words only
    with open(args.dict, "r", encoding="utf-8") as f:
        airline_words = [w.strip().lower() for w in f if w.strip()]

    # --- Make 10 random folds of 100 phrases each ---
    random.seed(42)
    n_folds = 3
    fold_size = 30
    folds = [random.sample(airline_words, fold_size) for _ in range(n_folds)]

    # Define search grid
    grid = {
        "α": [0.2, 0.3, 0.4, 0.5, 0.6],
        "β": [0, 0.05, 0.15, 0.25],
        "γ": [0, 0.05, 0.15, 0.25],
        "δ": [0, 0.05, 0.15, 0.25],
        "ε": [0, 0.05, 0.15, 0.25],
        "ζ": [0, 0.05, 0.15, 0.25],
    }

    # Generate all combinations
    all_combos = itertools.product(
        grid["α"], grid["β"], grid["γ"], grid["δ"], grid["ε"], grid["ζ"]
    )

    # Keep only those that sum to 1 (with tolerance for float error)
    valid_combos = [
        combo for combo in all_combos if abs(sum(combo) - 1.0) < 1e-6
    ]

    print("Number of valid combos:", len(valid_combos))

    best_params, best_score = None, -1.0
    for weights in valid_combos:   # <-- use only valid combos
        scores = []
        for fold in folds:
            win_rate, avg_guesses = evaluate_solver(solver, fold, weights)
            scores.append(win_rate)
        avg_score = sum(scores) / len(scores)

        print(f"weights={weights} -> avg win_rate={avg_score:.3f}")
        if avg_score > best_score:
            best_score = avg_score
            best_params = weights

    print("\nBest weights:", best_params, "with win_rate=", best_score)

if __name__ == "__main__":
    main()