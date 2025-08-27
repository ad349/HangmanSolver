# Hangman Solver 🎯

A Python-based Hangman solver designed for both **general English words** and **aviation/airline domain dictionaries**.  
It uses a mix of **information gain (EIG)**, **character priors**, **bigrams**, **position-based probabilities**, and **affix heuristics** to guess the next best letter.  

---

## Requirements
** You need to install nltk and download the ```words``` corpus**

## Features ✈️
- Supports **multi-word Hangman puzzles** (e.g., `"ancillary revenue"`).  
- Can combine the **general English dictionary (NLTK)** with a **custom airline/aviation dictionary**.  
- Uses:
  - Information gain (EIG) scoring.  
  - Position-based priors.  
  - Bigram context probabilities.  
  - Affix and orthographic heuristics (e.g., `ing`, `tion`, `air_`).  
- **Auto-play mode** to solve full words given the hidden target.  
- Falls back gracefully with open-vocabulary priors and a static frequency order.

---

## Solver Performance (Success Rate)

* Single Words : 77%
* Phrase (5-10 words) : 88%

---

## Methodology
A dictionary which has general english words and words from airlines domain is used to do pattern matching against, this dictionary was created by combing various online airlines sources (abbreviations, pdfs, common terms etc.) and words corpus from nltk. The model tries to pick the best guess by calculating a cumulative score of each candidate letter which provide maximim information gain and has highest context related probability. The weights were tuned by running grid search using n-fold cross validation on airlines domain specific words.

1. Dictionary Construction
The solver relies on a hybrid dictionary that combines:
  * General English words from the NLTK corpus.
  * Airline and aviation-specific terms compiled from various sources (PDFs, regulatory docs, common abbreviations, manuals, and domain-specific glossaries).
This ensures the solver can handle both everyday words and technical aviation vocabulary.

2. Candidate Filtering
For every partially revealed word (e.g., a__ro_la_e), the solver uses regex-based pattern matching to filter down possible candidates from the dictionary.
Words that conflict with already-guessed letters are removed.

3.Letter Scoring
For each candidate letter, multiple features are considered:
* Expected Information Gain (EIG): Chooses the letter that reduces the candidate set most effectively.
* Letter Priors: Frequency of a letter in the entire dictionary.
* Positional Priors: Likelihood of a letter appearing at a given position for words of similar length.
* Bigrams (Left/Right Context): Probability of a letter appearing next to already-known neighbors.
* Affix & Orthographic Heuristics: Proactive rules that reward letters that could complete common suffixes (-ing, -tion, -ed) or prefixes (pre-, dis-), as well as special cases (q → u, double consonants, etc.).

4. Weighted Scoring
Each feature contributes to a cumulative score, with weights tuned using grid search with n-fold cross validation on the aviation-specific dictionary.
The solver selects the letter with the highest combined score.

5. Fallbacks
If the word is out-of-vocabulary (OOV) (no candidates found), the solver falls back to:
Open-vocabulary priors (positional + bigrams + affix rules).
Static letter order (etaoin…) as the absolute last resort.

## Installation ⚙️
```bash
git clone https://github.com/yourusername/hangman-solver.git
cd hangman-solver
pip install -r requirements.txt
```
How to run the script?

1. Go to the working directory
2. There are 2 modes to run the script in -> auto or interactive. By default, the script runs in interactive mode - that is at each step you will have to provide the revealed letters in the hidden word / phrase. In auto mode, you provide the hidden word to the model and the model will output the guesses step by step and show final status (success or failed)
3. The dictionary is in the data folder which is required to run the script

### Example:

```python .\hangman_v4.py --dict <path-to-dictionary>```
Above, will launch the solver in interactivve mode. The process flows like below - 

```
Hangman Solver ready.
Example input (auto mode):
{"hiddenWord": "ancillary revenue", "currentWordState": "________ _______", "guessedLetters": [], "guessesRemaining": 6}
```

Your Input (Initial state of hidden word):
```
{"currentWordState": "________ _______", "guessedLetters": [], "guessesRemaining": 6}
```

The model's output is:
```{"nextGuess": "e", "status": "playing"}```

Your next input in case the letter guessed is in the hidden word will be:
```{"currentWordState": "________ _e_e__e", "guessedLetters": ["e"], "guessesRemaining": 6}```

If the guess is wrong, then:

```{"currentWordState": "________ _______", "guessedLetters": ["e"], "guessesRemaining": 5}```

### Launch in auto mode
```python .\hangman_v4.py --dict <path-to-dictionary> --auto```

```
Hangman Solver ready.
Example input (auto mode):
{"hiddenWord": "ancillary revenue", "currentWordState": "________ _______", "guessedLetters": [], "guessesRemaining": 6}
```

Your Input:
```
{"hiddenWord": "ancillary revenue", "currentWordState": "________ _______", "guessedLetters": [], "guessesRemaining": 6}
```

Output:
```
{"history": [{"pattern": "________  _e_e__", "guess": "e", "remaining": 6}, {"pattern": "_______r  re_e__", "guess": "r", "remaining": 6}, {"pattern": "_______r  re_e__", "guess": "s", "remaining": 5}, {"pattern": "_______r  re_e__", "guess": "t", "remaining": 4}, {"pattern": "____ll_r  re_e__", "guess": "l", "remaining": 4}, {"pattern": "____ll_r  re_e__", "guess": "p", "remaining": 3}, {"pattern": "_n__ll_r  re_en_", "guess": "n", "remaining": 3}, {"pattern": "_n__ll_r  re_enu", "guess": "u", "remaining": 3}, {"pattern": "an__llar  re_enu", "guess": "a", "remaining": 3}, {"pattern": "an__llar  re_enu", "guess": "d", "remaining": 2}, {"pattern": "an_illar  re_enu", "guess": "i", "remaining": 2}, {"pattern": "an_illar  revenu", "guess": "v", "remaining": 2}, {"pattern": "an_illar  revenu", "guess": "g", "remaining": 1}, {"pattern": "an_illar  revenu", "guess": "o", "remaining": 0}], "finalPattern": "an_illar  revenu", "hiddenWord": "ancillary revenue", "status": "failed"}
```

### To test the model against a dataset run ```test_model.py``` for single word and ```test_model-large.py``` for hidden phrases.
This script will return the summary stats on execution.

> You can edit test_model.py to change the words / phrases to test the solver on

### Example
```
PS C:\Users\USER\Desktop\IndigoProject> python .\test_model.py
Enter path to solver file: C:\Users\USER\Desktop\IndigoProject\hangman_v4.py
Word: ancillary            -> success in 9 guesses
Word: revenue              -> success in 7 guesses
Word: blackbox             -> failed in 10 guesses
Word: boarding             -> success in 11 guesses
Word: aircraft             -> success in 8 guesses
Word: passenger            -> success in 8 guesses
Word: cabin                -> failed in 8 guesses
Word: crew                 -> failed in 8 guesses
Word: baggage              -> failed in 9 guesses
Word: airport              -> success in 8 guesses
Word: terminal             -> success in 9 guesses
Word: fuel                 -> failed in 8 guesses
Word: runway               -> failed in 9 guesses
Word: delayed              -> success in 8 guesses
Word: flight               -> success in 11 guesses
Word: ticket               -> success in 10 guesses
Word: reservation          -> success in 9 guesses
Word: upgrade              -> success in 9 guesses
Word: economy              -> success in 9 guesses
Word: business             -> success in 10 guesses
Word: first                -> success in 10 guesses
Word: lounge               -> success in 9 guesses
Word: security             -> success in 10 guesses
Word: customs              -> success in 10 guesses
Word: arrival              -> success in 7 guesses
Word: departure            -> success in 7 guesses
Word: hub                  -> failed in 6 guesses
Word: route                -> success in 6 guesses
Word: schedule             -> success in 12 guesses
Word: pilot                -> success in 10 guesses
Word: engine               -> success in 4 guesses
Word: gate                 -> failed in 9 guesses
Word: overhead             -> success in 11 guesses
Word: checkin              -> success in 11 guesses
Word: carryon              -> success in 8 guesses
Word: regulation           -> success in 10 guesses
Word: compliance           -> success in 12 guesses
Word: network              -> success in 10 guesses
Word: alliance             -> success in 7 guesses
Word: operations           -> success in 9 guesses
Word: procedure            -> success in 7 guesses
Word: automation           -> success in 9 guesses
Word: safety               -> success in 11 guesses
Word: maintenance          -> success in 8 guesses
Word: ground               -> success in 11 guesses
Word: service              -> success in 9 guesses
Word: boarding pass        -> success in 15 guesses
Word: airline              -> success in 6 guesses
Word: aviation             -> success in 10 guesses
Word: weather              -> success in 10 guesses
Word: delay                -> success in 10 guesses
Word: cancellation         -> success in 9 guesses
Word: control              -> success in 11 guesses
Word: tower                -> failed in 10 guesses
Word: emergency            -> success in 8 guesses
Word: oxygen               -> success in 11 guesses
Word: mask                 -> failed in 9 guesses
Word: announcement         -> success in 9 guesses
Word: entertainment        -> success in 7 guesses
Word: wifi                 -> failed in 7 guesses
Word: seat                 -> success in 4 guesses
Word: belt                 -> failed in 9 guesses
Word: lifejacket           -> failed in 11 guesses
Word: turbulence           -> success in 11 guesses
Word: approach             -> success in 8 guesses
Word: landing              -> success in 10 guesses
Word: takeoff              -> failed in 9 guesses
Word: altitude             -> success in 8 guesses
Word: speed                -> success in 9 guesses
Word: navigation           -> success in 8 guesses
Word: radar                -> success in 4 guesses
Word: cockpit              -> failed in 7 guesses
Word: training             -> success in 8 guesses
Word: simulator            -> success in 10 guesses
Word: charter              -> success in 9 guesses
Word: cargo                -> failed in 8 guesses
Word: logistics            -> success in 10 guesses
Word: fleet                -> success in 8 guesses
Word: leasing              -> success in 8 guesses
Word: partnership          -> success in 9 guesses
Word: subsidiary           -> success in 9 guesses
Word: market               -> failed in 10 guesses
Word: competition          -> success in 11 guesses
Word: allotment            -> success in 10 guesses
Word: ancillary revenue    -> success in 11 guesses
Word: on time              -> failed in 10 guesses
Word: delayed flight       -> success in 15 guesses
Word: ground handling      -> success in 12 guesses
Word: cabin crew           -> success in 13 guesses
Word: check in             -> failed in 10 guesses
Word: frequent flyer       -> success in 12 guesses
Word: jet bridge           -> failed in 9 guesses
Word: boarding gate        -> success in 12 guesses
Word: flight schedule      -> failed in 13 guesses
Word: safety procedure     -> success in 15 guesses
Word: passenger service    -> success in 12 guesses
Word: ground damage        -> success in 12 guesses
Word: out of service       -> failed in 15 guesses
Word: flight reroute       -> success in 14 guesses
Word: oversized baggage    -> success in 15 guesses
Word: overhead compartment -> success in 15 guesses
Word: life vest            -> failed in 11 guesses
Word: touch down           -> failed in 13 guesses
Word: minimum equipment list -> success in 13 guesses
Word: base maintenance     -> success in 12 guesses

=== SUMMARY ===
{
  "totalWords": 105,
  "successes": 81,
  "failures": 24,
  "successRate": 0.7714285714285715,
  "avgGuesses": 9.666666666666666,
  "avgWrongGuesses": 3.3523809523809525
}
```



