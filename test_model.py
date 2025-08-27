import subprocess
import json
import statistics
import sys

# Example test set (extend to 100+ words)
TEST_WORDS = [
    "ancillary", "revenue", "blackbox", "boarding", "aircraft", "passenger",
    "cabin", "crew", "baggage", "airport", "terminal", "fuel", "runway",
    "delayed", "flight", "ticket", "reservation", "upgrade", "economy",
    "business", "first", "lounge", "security", "customs", "arrival", "departure",
    "hub", "route", "schedule", "pilot", "engine", "gate", "overhead", "checkin",
    "carryon", "regulation", "compliance", "network", "alliance", "operations",
    "procedure", "automation", "safety", "maintenance", "ground", "service",
    "boarding pass", "airline", "aviation", "weather", "delay", "cancellation",
    "control", "tower", "emergency", "oxygen", "mask", "announcement",
    "entertainment", "wifi", "seat", "belt", "lifejacket", "turbulence",
    "approach", "landing", "takeoff", "altitude", "speed", "navigation",
    "radar", "cockpit", "training", "simulator", "charter", "cargo",
    "logistics", "fleet", "leasing", "partnership", "subsidiary", "market",
    "competition", "allotment", "ancillary revenue", "on time", "delayed flight",
    "ground handling", "cabin crew", "check in", "frequent flyer", "jet bridge",
    "boarding gate", "flight schedule", "safety procedure", "passenger service",
    "ground damage", "out of service", "flight reroute", "oversized baggage",
    "overhead compartment", "life vest", "touch down", "minimum equipment list",
    "base maintenance"
]

def run_solver(word, solver_path):
    # Build initial masked state
    pattern = "".join("_" if c != " " else " " for c in word)
    input_json = {
        "hiddenWord": word,
        "currentWordState": pattern,
        "guessedLetters": [],
        "guessesRemaining": 6
    }

    # Run solver in auto mode
    proc = subprocess.Popen(
        ["python", solver_path, "--auto"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = proc.communicate(json.dumps(input_json) + "\n")
    if err:
        print("Error:", err)
    try:
        return json.loads(out.strip().split("\n")[-1])  # last JSON line
    except Exception as e:
        print("Parse error:", out)
        return None

def summarize(results):
    total = len(results)
    successes = sum(1 for r in results if r and r["status"] == "success")
    avg_guesses = statistics.mean(
        len(r["history"]) for r in results if r
    )
    wrong_guesses = statistics.mean(
        sum(1 for h in r["history"] if h["guess"] not in r["hiddenWord"]) 
        for r in results if r
    )
    return {
        "totalWords": total,
        "successes": successes,
        "failures": total - successes,
        "successRate": successes / total,
        "avgGuesses": avg_guesses,
        "avgWrongGuesses": wrong_guesses
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        solver_path = input("Enter path to solver file: ").strip()
    else:
        solver_path = sys.argv[1]

    results = []
    for w in TEST_WORDS:
        res = run_solver(w, solver_path)
        results.append(res)
        if res:
            print(f"Word: {w:20} -> {res['status']} in {len(res['history'])} guesses")
        else:
            print(f"Word: {w:20} -> ERROR")

    stats = summarize(results)
    print("\n=== SUMMARY ===")
    print(json.dumps(stats, indent=2))