# Hangman Solver 🎯

A Python-based Hangman solver designed for both **general English words** and **aviation/airline domain dictionaries**.  
It uses a mix of **information gain (EIG)**, **character priors**, **bigrams**, **position-based probabilities**, and **affix heuristics** to guess the next best letter.  

---

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

The model's output is ```{"nextGuess": "e", "status": "playing"}```

Your next input in case the letter guessed is in the hidden word will be:
```{"currentWordState": "________ _e_e__e", "guessedLetters": ["e"], "guessesRemaining": 6}```

If the guess is wrong, then:

```{"currentWordState": "________ _______", "guessedLetters": ["e"], "guessesRemaining": 5}```
