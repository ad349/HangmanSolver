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

How to run the script?

Go to the working directory
>IndigoProject

python .\hangman_v4.py
>
Hangman Solver ready.
Example input (auto mode):
{"hiddenWord": "ancillary revenue", "currentWordState": "________ _______", "guessedLetters": [], "guessesRemaining": 6}```
