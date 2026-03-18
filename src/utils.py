import os
import json
import matplotlib.pyplot as plt

def setup_dirs():
    """
    Creates the necessary directories if they don't exist.
    """
    dirs = ["results", "logs"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_json(obj, path):
    """
    Saves a Python object as a JSON file.
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
    print(f"Saved JSON to {path}")

def plot(values, path, title="Plot"):
    """
    Plots a list of values and saves the figure.
    """
    plt.figure(figsize=(8,5))
    plt.plot(values, marker='o', markersize=3)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Plot saved to {path}")