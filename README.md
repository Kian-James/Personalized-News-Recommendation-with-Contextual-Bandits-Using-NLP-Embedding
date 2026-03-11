# Personalized News Recommendation (Stream Data Based)  

[![Python Version](https://img.shields.io/badge/python->=2.7.13-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

---

## 🚀 Objective
Study reinforcement learning approaches for building an article recommender system.  
The project focuses on interactive recommendation using **contextual multi-armed bandits** to learn user preferences and suggest new articles while receiving feedback.

---

## 📝 Problem
Interactive recommendation is modeled as a **contextual multi-armed bandit problem**, addressing:  

- **Exploitation vs. exploration trade-off**  
- **Cold-start problem** for new users or articles  

---

## 💡 Solution
We explore:  

1. **Content-based recommendations**  
2. **Collaborative filtering recommendations**  

Algorithms implemented and compared:  
- **LinUCB**  
- **Hybrid LinUCB**  
- **ε-Greedy**  
- **Random selection**  

Data preprocessing generates user and article feature vectors for model input.

---

## ⚙️ Technical Details

### Algorithms

**LinUCB**  
- Selects arms using upper confidence bounds with context vectors.  
- Assumes user/bandit independence and only observed features.

**Hybrid LinUCB**  
- Incorporates **user-article interactions**, capturing dependencies.

**ε-Greedy**  
- Exploits best-known options most of the time (`1-ε`) and explores randomly (`ε`).  
- Does not use user-article information.  
- Typical ε = 0.1  

**Random**  
- Selects articles randomly. No learning occurs over time.

---

## 🧰 Setup & Usage

### Requirements
```bash
python >= 2.7.13
pip install -r requirements.txt
