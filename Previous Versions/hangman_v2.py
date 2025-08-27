import re
import string
from collections import Counter, defaultdict
import nltk
from nltk.corpus import words as nltk_words
import json
import sys
import argparse

class HangmanSolver:
    def __init__(self, airline_dict_path=None, multiword_mode=True):
        # Load nltk words (general English dictionary)
        self.word_list = {w.lower() for w in nltk_words.words() if w.isalpha()}

        # Merge airline dictionary if provided
        if airline_dict_path:
            with open(airline_dict_path, "r", encoding="utf-8") as f:
                airline_words = {w.strip().lower() for w in f if w.strip()}
            self.word_list |= airline_words  

        self.word_list = list(self.word_list)
        self.multiword_mode = multiword_mode

        # Priors (safe defaults for OOV scoring)
        self.letter_prior = {c: (1.0/26) for c in string.ascii_lowercase}
        self.pos_prior = {}
        self.left_bigram = {}
        self.right_bigram = {}

        self.reset()

    def reset(self):
        self.guessedLetters = set()
        self.currentWordState = ""
        self.guessesRemaining = 6

    def _build_priors(self, words):
        self.letters = [chr(c) for c in range(ord('a'), ord('z')+1)]
        # Global letter prior
        self.letter_prior = Counter()
        # Positional prior: length -> pos -> letter -> count
        self.pos_prior = defaultdict(lambda: defaultdict(Counter))
        # Bigram priors
        self.left_bigram = defaultdict(Counter)   # P(l | left)
        self.right_bigram = defaultdict(Counter)  # P(right | l)

        for w in words:
            for i, ch in enumerate(w):
                self.letter_prior[ch] += 1
                self.pos_prior[len(w)][i][ch] += 1
                if i > 0:
                    self.left_bigram[w[i-1]][ch] += 1
                if i < len(w)-1:
                    self.right_bigram[ch][w[i+1]] += 1

        # Normalize-ish helpers
        def norm(counter):
            total = sum(counter.values()) or 1
            return {k: v/total for k, v in counter.items()}

        self.letter_prior = norm(self.letter_prior)
        # For pos_prior and bigrams we’ll look up with smoothing on demand.

    # ---------- Candidate machinery ----------
    def filter_candidates_one_word(self, word_pattern, guessed):
        """Return candidates for ONE word-pattern like '_la_k_o_' (no spaces)."""
        regex = "^" + word_pattern.replace("_", "[a-z]") + "$"
        blanks_idx = [i for i, ch in enumerate(word_pattern) if ch == "_"]
        fixed_idx = [(i, ch) for i, ch in enumerate(word_pattern) if ch != "_"]

        cands = []
        for w in self.word_list:
            if len(w) != len(word_pattern):
                continue
            if not re.match(regex, w):
                continue
            # reject words containing letters known to be absent anywhere
            # (i.e., letters guessed but not present in revealed pattern)
            if any(g in w for g in guessed if g not in word_pattern):
                continue
            # enforce fixed positions explicitly (redundant with regex, but cheap)
            ok = True
            for i, ch in fixed_idx:
                if w[i] != ch:
                    ok = False
                    break
            if not ok:
                continue
            cands.append(w)
        # print(cands)
        return cands
    
    def _split_state(self, currentWordState):
        # Preserve spaces; split into words
        return [w for w in currentWordState.lower().split(" ") if w != ""]

    # ---------- EIG tie-break on most constrained word ----------
    def _eig_letter_for_word(self, pattern, candidates, guessed):
        """Pick letter minimizing expected remaining candidates on this word."""
        blanks_idx = [i for i, ch in enumerate(pattern) if ch == "_"]
        if not blanks_idx or not candidates:
            return None

        best_letter, best_expect = None, float("inf")
        already = set(guessed)

        # letter universe limited to letters that actually appear in candidates OR good fallbacks
        letter_pool = set()
        for w in candidates:
            for ch in set(w):
                if ch not in guessed:
                    letter_pool.add(ch)
        if not letter_pool:
            letter_pool = set([chr(c) for c in range(ord('a'), ord('z')+1)]) - already
        # print(letter_pool)

        total = len(candidates)
        # print(total)
        for l in letter_pool:
            # Partition candidates by the "reveal mask" for l
            buckets = Counter()
            for w in candidates:
                mask_bits = []
                for i in blanks_idx:
                    mask_bits.append('1' if w[i] == l else '0')
                mask_key = "".join(mask_bits)  # e.g., "0100"
                # print(mask_key)
                buckets[mask_key] += 1
            # Expected remaining size after guessing l:
            # E[size] = sum_i (p_i * size_i) with p_i = size_i/total = sum(size_i^2)/total
            # print(buckets)
            # print(buckets.values())
            expected_remaining = sum(sz*sz for sz in buckets.values()) / total
            # print(expected_remaining)
            if expected_remaining < best_expect:
                best_expect = expected_remaining
                best_letter = l
                # print(best_letter)

        return best_letter

    # ---------- Smarter OOV fallback ----------
    def _oov_score_letter_for_phrase(self, words_state, guessed):
        """
        When no candidates exist, score next letter by priors + affix & orthographic heuristics.
        """
        alpha, beta, gamma = 0.25, 0.45, 0.30
        best_letter, best_score = None, -1.0
        remaining_letters = set(string.ascii_lowercase) - set(guessed)

        def affix_bonus(pattern, letter):
            score = 0
            # suffixes
            if pattern.endswith("___"):
                if letter in "ing":
                    score += 5
            if pattern.endswith("__"):
                if letter in "edr":
                    score += 4
            if pattern.endswith("____"):
                if letter in "tionment":
                    score += 6
            if pattern.endswith("_ity") and letter in "ity":
                score += 4
            if pattern.endswith("_ous") and letter in "ous":
                score += 4
            # prefixes
            if pattern.startswith("___") and letter in "preproreunindis":
                score += 3
            # orthographic rules
            if "q" in pattern and letter == "u":
                score += 8
            if "c" in pattern and letter == "k":
                score += 5
            # double consonants
            for i in range(1, len(pattern) - 1):
                if pattern[i] == "_" and pattern[i-1] == "_":
                    if letter in "lnrst":
                        score += 3
            return score

        for l in remaining_letters:
            score = 0.0
            for w in words_state:
                for i, ch in enumerate(w):
                    if ch != "_":
                        continue
                    left = w[i-1] if i > 0 else "^"
                    right = w[i+1] if i < len(w)-1 else "$"

                    lp = self.letter_prior.get(l, 0.0)
                    pp = self.pos_prior.get((i, l), 0.01)
                    lb = self.left_bigram.get((left, l), 0.01)
                    rb = self.right_bigram.get((l, right), 0.01)

                    score += alpha*lp + beta*pp + gamma*(lb + rb)
                    score += affix_bonus(w, l)

            if score > best_score:
                best_score = score
                best_letter = l

        return best_letter

     # ---------- Public: next guess ----------
    def get_next_guess(self, currentWordState, guessedLetters, guessesRemaining):
        guessed = {g for g in (ch.lower().strip() for ch in guessedLetters) if len(g) == 1 and g.isalpha()}
        words_state = self._split_state(currentWordState)

        # Build candidate sets per word
        per_word = []
        for wpat in words_state:
            cands = self.filter_candidates_one_word(wpat, guessed)
            per_word.append((wpat, cands))

        # 1) If any word has exactly ONE candidate, force its missing letter
        for wpat, cands in per_word:
            if len(cands) == 1:
                for ch in cands[0]:
                    if ch not in guessed and "_" in wpat:
                        # ensure it's actually filling a blank
                        return {"nextGuess": ch, "status": "playing"}

        # 2) If we have ANY candidates, use EIG on the most constrained word
        constrained = [(wpat, cands) for (wpat, cands) in per_word if len(cands) > 0 and "_" in wpat]

        if constrained:
            # choose the word with the fewest candidates
            wpat, cands = min(constrained, key=lambda t: len(t[1]))

            eig_letter = self._eig_letter_for_word(wpat, cands, guessed)
            if eig_letter and eig_letter not in guessed:
                return {"nextGuess": eig_letter, "status": "playing"}

            # fallback to frequency within ALL candidates if eig returns None
            freq = Counter()
            for _, cset in constrained:
                for w in cset:
                    for ch in set(w):
                        if ch not in guessed:
                            freq[ch] += 1
            if freq:
                return {"nextGuess": freq.most_common(1)[0][0], "status": "playing"}

        # 3) OOV fallback: open-vocab priors (positional + bigrams + global)
        oov_letter = self._oov_score_letter_for_phrase(words_state, guessed)
        if oov_letter:
            return {"nextGuess": oov_letter, "status": "playing"}

        # 4) Absolute last resort: static order
        for ch in "etaoinrshlcdumpgbyfvkwzxq":
            if ch not in guessed:
                return {"nextGuess": ch, "status": "playing"}

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