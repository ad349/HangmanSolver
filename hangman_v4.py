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

        self._build_priors(self.word_list)
        self.reset()

    def reset(self):
        self.guessedLetters = set()
        self.currentWordState = ""
        self.guessesRemaining = 6

    def _build_priors(self, words):
        self.letters = [chr(c) for c in range(ord('a'), ord('z')+1)]
        self.letter_prior = Counter()
        self.pos_prior = defaultdict(lambda: defaultdict(Counter))
        self.left_bigram = defaultdict(Counter)
        self.right_bigram = defaultdict(Counter)
    
        for w in words:
            for i, ch in enumerate(w):
                self.letter_prior[ch] += 1
                self.pos_prior[len(w)][i][ch] += 1
                if i > 0:
                    self.left_bigram[w[i-1]][ch] += 1
                else:
                    self.left_bigram["^"][ch] += 1   # <-- start boundary
                if i < len(w)-1:
                    self.right_bigram[ch][w[i+1]] += 1
                else:
                    self.right_bigram[ch]["$"] += 1  # <-- end boundary
    
        def norm(counter):
            total = sum(counter.values()) or 1
            return {k: v/total for k, v in counter.items()}
    
        # Normalize everything
        self.letter_prior = norm(self.letter_prior)
    
        self.pos_prior = {
            length: {
                pos: norm(counter) for pos, counter in pos_dict.items()
            } for length, pos_dict in self.pos_prior.items()
        }
    
        self.left_bigram = {left: norm(counter) for left, counter in self.left_bigram.items()}
        self.right_bigram = {ch: norm(counter) for ch, counter in self.right_bigram.items()}

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
        """Pick letter using EIG + priors + affix/orthographic heuristics."""
        blanks_idx = [i for i, ch in enumerate(pattern) if ch == "_"]
        if not blanks_idx or not candidates:
            return None

        remaining_letters = set(ch for w in candidates for ch in w) - guessed
        if not remaining_letters:
            remaining_letters = set(self.letters) - guessed
        # print(pattern)
        # print(remaining_letters)
        best_letter, best_score = None, -1.0
        total = len(candidates)

        # Weights – tune later
        alpha, beta, gamma, delta, epsilon, eta = 0.4, 0.25, 0.15, 0.05, 0, 0.15

        for l in remaining_letters:
            # --- EIG ---
            buckets = Counter()
            for w in candidates:
                mask_bits = ["1" if w[i] == l else "0" for i in blanks_idx]
                buckets["".join(mask_bits)] += 1
            expected_remaining = sum(sz * sz for sz in buckets.values()) / total
            eig_score = 1.0 - (expected_remaining / total)  # normalize so bigger = better
            # print(eig_score, expected_remaining, total)
            # --- Priors ---
            lp = self.letter_prior.get(l, 0.0)
            # print(lp)
            pos_score = 0.0
            for i in blanks_idx:
                pos_score += self.pos_prior[len(pattern)][i].get(l, 0.0)
            # print(pos_score)
            left_bigram_score = 0.0
            right_bigram_score = 0.0
            for i in blanks_idx:
                left = pattern[i-1] if i > 0 and pattern[i-1] != "_" else "^"
                right = pattern[i+1] if i < len(pattern)-1 and pattern[i+1] != "_" else "$"
                left_bigram_score += self.left_bigram[left].get(l, 0.0)
                right_bigram_score += self.right_bigram[l].get(right, 0.0)
            # print(left, right)
            affix_score = self._affix_bonus(l, pattern)

            # --- Combined ---
            score = alpha*eig_score + beta*lp + gamma*pos_score + delta*left_bigram_score + epsilon*right_bigram_score + eta*affix_score
            # print("letter-> ", l, "scores ->", score, eig_score, lp, pos_score, left_bigram_score, right_bigram_score, affix_score)

            if score > best_score:
                best_score, best_letter = score, l
        # print(best_letter)
        return best_letter

    # New helper for affix / orthographic bonus
    def _affix_bonus(self, letter, pattern):
        """
        Proactive + reactive affix/orthographic bonus, normalized to [0,1].
        Rewards letters that would COMPLETE common affixes given the current
        pattern with blanks. Also keeps a few reactive rules (like 'q' -> 'u').
        """
        b = 0
    
        s = pattern  # alias
    
        # -------- Proactive suffix completions --------
        # ...ing
        if len(s) >= 3:
            tail3 = s[-3:]
            if tail3 == "i__":
                if letter == "n": b += 6   # i__ -> i n _  (toward "ing")
                if letter == "o": b += 3   # i__ -> i o _  (toward "ion")
            if tail3 == "in_":
                if letter == "g": b += 7   # in_ -> ing
    
            # ...ion
            if tail3 == "_io":
                if letter == "n": b += 6   # _io -> n (..nion)
            if tail3 == "ti_":
                if letter == "o": b += 5   # ti_ -> tio (..tion)
            if tail3 == "tio":
                if letter == "n": b += 6   # tio -> tion
    
            # ...ed
            if tail3 == "__d":
                if letter == "e": b += 4   # __d -> _ e d
            if tail3 == "_ed":
                # often consonant before -ed
                if letter in "trnsl": b += 3
    
            # ...er
            if tail3 == "_er":
                if letter in "tnrlds": b += 3
    
            # ...ly
            if tail3 == "_ly":
                if letter in "ble": b += 3
    
            # ...ous
            if tail3 == "ou_":
                if letter == "s": b += 5
    
            # ...ment
            if tail3 == "me_":
                if letter == "n": b += 4
            if len(s) >= 4 and s[-4:] == "men_":
                if letter == "t": b += 5
    
            # ...able
            if tail3 == "ab_":
                if letter == "l": b += 4
            if len(s) >= 4 and s[-4:] == "abl_":
                if letter == "e": b += 4
    
            # ...ive
            if tail3 == "iv_":
                if letter == "e": b += 4
    
        # -------- Proactive prefix completions --------
        if len(s) >= 3:
            head3 = s[:3]
            if head3 == "pre":
                if len(s) > 3 and s[3] == "_":
                    if letter in "smt": b += 3  # pre_...
            if head3 == "dis":
                if len(s) > 3 and s[3] == "_":
                    if letter in "cpt": b += 3  # dis_...
            if head3 == "uni":
                if len(s) > 3 and s[3] == "_":
                    if letter in "tfn": b += 2  # uni_...
    
        # Aviation-ish common chunks (light, generic heuristics)
        # avio(n), air_, aero_, auto_
        if len(s) >= 3 and s[-3:] == "avi":
            if letter == "o": b += 3
        if s.startswith("air") and len(s) > 3 and s[3] == "_":
            if letter in "cfst": b += 2
        if s.startswith("aer") and len(s) > 3 and s[3] == "_":
            if letter in "od": b += 2
    
        # -------- Orthographic / reactive helpers --------
        if "q" in s and letter == "u":
            b += 7
        # Encourage double consonants when we see "__" (somewhere)
        if "__" in s and letter in "lnrstm":
            b += 3
    
        # If the last revealed char is a consonant and end is '_' (C_), double?
        if len(s) >= 2 and s[-1] == "_" and s[-2] != "_" and s[-2] not in "aeiou":
            if letter == s[-2]:  # e.g., ...n_ -> suggest 'n'
                b += 3
    
        # -------- Normalize --------
        MAX_BONUS = 10.0
        if b < 0: b = 0
        if b > MAX_BONUS: b = MAX_BONUS
        return b / MAX_BONUS
    
    # ---------- Smarter OOV fallback ----------
    def _oov_score_letter_for_phrase(self, words_state, guessed):
        remaining_letters = set(self.letters) - guessed
        best_letter, best_score = None, -1.0
    
        alpha, beta, gamma, delta, epsilon, eta = 0.0, 0.2, 0.3, 0.2, 0.2, 0.1  # no EIG when OOV
    
        for l in remaining_letters:
            score = 0.0
            for w in words_state:
                for i, ch in enumerate(w):
                    if ch != "_":
                        continue
    
                    # replace "_" neighbors with boundary markers
                    left = w[i-1] if i > 0 and w[i-1] != "_" else "^"
                    right = w[i+1] if i < len(w)-1 and w[i+1] != "_" else "$"
    
                    lp = self.letter_prior.get(l, 0.0)
                    pp = self.pos_prior[len(w)][i].get(l, 0.0)
                    lb = self.left_bigram.get(left, {}).get(l, 0.0)
                    rb = self.right_bigram.get(l, {}).get(right, 0.0)
    
                    score += beta*lp + gamma*pp + gamma*lb + epsilon*rb
    
                score += eta * self._affix_bonus(l, w)
    
            if score > best_score:
                best_score, best_letter = score, l
    
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
                # print("one candidate left ", cands)
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
        default="C:\\Users\\USER\\Desktop\\IndigoProject\\data\\airlines_unique_words.txt",
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