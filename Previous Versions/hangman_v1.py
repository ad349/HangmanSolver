import re
import string
from collections import Counter
import nltk
from nltk.corpus import words as nltk_words
import json
import sys
import argparse

class HangmanSolver:
    def __init__(self, airline_dict_path=None):
        self.word_list = {w.lower() for w in nltk_words.words() if w.isalpha()}

        if airline_dict_path:
            with open(airline_dict_path, "r", encoding="utf-8") as f:
                airline_words = {w.strip().lower() for w in f if w.strip()}
            self.word_list |= airline_words  

        self.word_list = list(self.word_list)
        self.reset()

    def reset(self):
        self.guessedLetters = set()
        self.currentWordState = ""
        self.guessesRemaining = 6

    def filter_candidates(self, pattern, guessed):
        regex = "^" + pattern.replace("_", "[a-z]") + "$"
        candidates = []
        for word in self.word_list:
            if len(word) != len(pattern):
                continue
            if re.match(regex, word):
                if any(letter in word for letter in guessed if letter not in pattern):
                    continue
                candidates.append(word)
        return candidates

    def get_next_guess(self, currentWordState, guessedLetters, guessesRemaining):
        self.currentWordState = currentWordState
        self.guessedLetters = set(guessedLetters)
        self.guessesRemaining = guessesRemaining

        words_state = currentWordState.lower().split(" ")

        all_candidates = []
        for state in words_state:
            if not state:
                continue
            candidates = self.filter_candidates(state, self.guessedLetters)
            all_candidates.append(candidates)
        # print(all_candidates)
        print([len(candidate) for candidate in all_candidates])
        print([candidate for candidate in all_candidates])
        
        for candidates in all_candidates:
            if len(candidates) == 1:
                for letter in candidates[0]:
                    if letter not in self.guessedLetters:
                        return {"nextGuess": letter, "status": "playing"}

        counts = Counter()
        for candidates in all_candidates:
            for word in candidates:
                for letter in set(word):
                    if letter not in self.guessedLetters:
                        counts[letter] += 1

        if counts:
            return {"nextGuess": counts.most_common(1)[0][0], "status": "playing"}

        fallback_order = "etaoinrslcdupmghbyfvkwzxq"
        for letter in fallback_order:
            if letter not in self.guessedLetters:
                return {"nextGuess": letter, "status": "playing"}

        return {"nextGuess": "", "status": "reset"}


def update_pattern(hidden_word, current_pattern, guess):
    """Reveals guessed letters in the current pattern based on the hidden word."""
    new_pattern = []
    for hw_char, pat_char in zip(hidden_word, current_pattern):
        if hw_char == " ":
            new_pattern.append(" ")
        elif pat_char != "_":
            new_pattern.append(pat_char)
        elif hw_char.lower() == guess.lower():
            new_pattern.append(hw_char.lower())
        else:
            new_pattern.append("_")
    return "".join(new_pattern)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Enable auto-play until solved or guesses run out (requires hiddenWord in input JSON)"
    )
    parser.add_argument(
        "--dict", "-d",
        type=str,
        default="C:\\Users\\USER\\Desktop\\indigo\\data\\airlines_unique_words.txt",
        help="Path to airline dictionary"
    )
    args = parser.parse_args()

    solver = HangmanSolver(airline_dict_path=args.dict)

    print("Hangman Solver ready.")
    print("Example input (auto mode):")
    print("{\"hiddenWord\": \"ancillary revenue\", \"currentWordState\": \"________ _______\", \"guessedLetters\": [], \"guessesRemaining\": 6}")

    for line in sys.stdin:
        try:
            input_json = json.loads(line.strip())
            pattern = input_json["currentWordState"]
            guessesRemaining = input_json["guessesRemaining"]
            guessedLetters = input_json["guessedLetters"]

            if not args.auto:
                if "_" not in pattern or guessesRemaining <= 0:
                    print(json.dumps({"nextGuess": "", "status": "reset"}))
                    continue
                output = solver.get_next_guess(pattern, guessedLetters, guessesRemaining)
                print(json.dumps(output))

            else:
                hidden_word = input_json.get("hiddenWord")
                if not hidden_word:
                    print(json.dumps({"error": "hiddenWord required in auto mode"}))
                    continue

                history = []
                while "_" in pattern and guessesRemaining > 0:
                    output = solver.get_next_guess(pattern, guessedLetters, guessesRemaining)
                    guess = output["nextGuess"]

                    if not guess:
                        history.append({"nextGuess": "", "status": "reset"})
                        break

                    guessedLetters.append(guess)

                    if guess in hidden_word.lower():
                        pattern = update_pattern(hidden_word.lower(), pattern, guess)
                    else:
                        guessesRemaining -= 1

                    history.append({"pattern": pattern, "guess": guess, "remaining": guessesRemaining})

                print(json.dumps({
                    "history": history,
                    "finalPattern": pattern,
                    "hiddenWord": hidden_word,
                    "status": "success" if "_" not in pattern else "failed"
                }))

        except Exception as e:
            print(json.dumps({"error": str(e)}))