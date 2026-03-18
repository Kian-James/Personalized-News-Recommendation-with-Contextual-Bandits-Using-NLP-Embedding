import numpy as np

def dcg(relevance):
    return np.sum([(2**rel - 1) / np.log2(i+2) for i, rel in enumerate(relevance)])

def ndcg_at_k(true_label, ranked_list, k=5):
    relevance = [1 if i == true_label else 0 for i in ranked_list[:k]]
    ideal = sorted(relevance, reverse=True)
    return dcg(relevance) / (dcg(ideal) + 1e-9)

def hit_at_k(true_label, ranked_list, k=5):
    return 1 if true_label in ranked_list[:k] else 0