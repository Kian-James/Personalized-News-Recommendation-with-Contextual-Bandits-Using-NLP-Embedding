# ⚖️ Ethics Statement — Personalized News Recommendation with Contextual Bandits

> **Project:** Personalized News Recommendation with Contextual Bandits Using NLP Embedding  
> **Institution:** Holy Angel University, Angeles City, Pampanga  
> **Authors:** Jero E. Halili, Kian Andrei G. James, Gerail C. Mendoza, Angelica Z. Tinio  
> **Version:** 1.0

---

## 1. Purpose of This Statement

This document outlines the ethical considerations, risks, and mitigation strategies associated with the design, training, evaluation, and potential deployment of a personalized news recommendation system. We believe that responsible AI development requires naming risks explicitly — not as disclaimers, but as design constraints.

---

## 2. Filter Bubbles and Viewpoint Diversity

### Risk
Recommendation systems that optimize for engagement (clicks) tend to reinforce existing preferences. Over time, a user receives articles increasingly similar to what they already believe — a phenomenon known as a **filter bubble** or **echo chamber effect**. This is a well-documented risk in algorithmic content curation (Pariser, 2011).

In our system, the contextual bandit agent rewards clicks with `+1` and non-clicks with `0`. This binary reward structure has no mechanism to value **diversity of perspective**, meaning the agent may converge on a narrow content profile per user even when broader exposure would be more socially beneficial.

### Mitigation
- Diversity constraints are recommended in the ranking algorithm to ensure recommendations span multiple categories from the 42 available in the dataset, rather than concentrating on a user's dominant interest cluster
- Future iterations should explore **diversity-aware reward shaping** that adds a small positive signal for recommending articles from underrepresented categories in a user's recent history
- Periodic injection of out-of-distribution articles (deliberate exploration) can expose users to new topics without fully abandoning relevance

---

## 3. Data Privacy

### Risk
Recommendation systems that learn from user behavior inherently process personal preference data. Mishandled interaction logs can expose reading habits, political views, health interests, or other sensitive behavioral patterns.

### Current Safeguards
- **This project uses no real user data.** All user interactions are simulated during offline evaluation
- The training dataset (Misra, 2022) contains only published article metadata — headlines, descriptions, categories, authors, and dates — with **no user-identifying information**
- The dataset is publicly available on Kaggle under an open research license

### Future Deployment Requirements
Any extension of this system that collects real user interaction data must:
- Comply with applicable data protection laws (e.g., the Philippine Data Privacy Act of 2012, Republic Act 10173; GDPR for international deployments)
- Rely exclusively on **anonymized interaction logs** with no linkage to user identity
- Implement **data minimization** — collecting only what is necessary for model improvement
- Provide users with clear opt-out mechanisms and data deletion rights
- Undergo a Data Privacy Impact Assessment (DPIA) before deployment

---

## 4. Dataset Bias

### Risk
The News Category Dataset consists exclusively of HuffPost articles published between 2012 and 2022. This introduces several forms of bias:

| Bias Type | Description |
|---|---|
| **Source bias** | Single publisher perspective; HuffPost has a known editorial leaning |
| **Temporal bias** | Topics from 2012–2022 may not reflect current events, language, or cultural context |
| **Category imbalance** | Some categories (e.g., Politics, Entertainment) have far more articles than others (e.g., Environment, Sports) |
| **Geographic bias** | Content is predominantly U.S.-centric; international perspectives may be underrepresented |
| **Linguistic bias** | All articles are in English; non-English speakers are excluded from optimal performance |

### Mitigation
- Oversampling of underrepresented categories was applied during data pipeline preprocessing
- Fairness-aware evaluation measures are recommended to track per-category performance separately, not only aggregate accuracy
- Future versions should incorporate diverse, multilingual news sources to improve generalizability

---

## 5. Reward Function Bias

### Risk
The current reward function assigns `+1` for a simulated click and `0` for no click. This is a **proxy metric** — it measures predicted engagement, not actual information quality, credibility, or user benefit.

This can unintentionally favor:
- **Sensational or emotionally charged headlines** over substantive reporting
- **High-visibility topics** over important but less clickable content
- **Familiar content** over challenging or educational material

This mirrors real-world problems documented in social media recommendation research, where engagement optimization has been linked to the amplification of misinformation and outrage-inducing content.

### Mitigation
- Incorporate **multi-objective reward signals** in future versions (e.g., dwell time, article completion rate, explicit ratings)
- Consider adding a **credibility signal** based on source quality or fact-checking metadata
- Regular audits of recommended content distributions across categories can surface unintended reward-driven biases

---

## 6. Algorithmic Transparency and Explainability

### Risk
The recommendation pipeline uses Sentence Transformer embeddings (dense, 384-dimensional vectors) and epsilon-greedy or LinUCB bandit agents. These components are not inherently interpretable — a user or stakeholder cannot easily understand **why** a specific article was recommended.

Lack of explainability creates accountability gaps: if a system recommends harmful or biased content, it becomes difficult to identify the cause and correct it.

### Mitigation
- Future iterations should implement **explainability mechanisms**, such as:
  - Displaying the top keywords or topics that influenced a recommendation
  - Showing similarity scores between recommended articles and the user's recent reading history
  - Providing a brief rationale (e.g., "Recommended because you read articles about Climate Change")
- Attention-based models (e.g., NRMS with multi-head self-attention) offer more interpretable intermediate representations and should be explored in future versions

---

## 7. Misuse Risks

### Risk
A working news recommendation system could be misused to:
- **Microtarget** users with politically or commercially motivated content
- **Suppress** exposure to counter-narratives or fact-checks
- **Manipulate** public opinion by exploiting filter bubbles at scale

### Position
This project is an academic prototype evaluated exclusively in an offline simulation environment. It is **not** designed or intended for use as a political influence tool, commercial targeting system, or any application that manipulates information access without user awareness and consent.

The authors explicitly oppose any deployment of this system — or derivative works — for purposes of information manipulation, disinformation, or covert behavioral influence.

---

## 8. Generalizability and Scope Limitations

- Results are valid only within the offline simulation setting using the HuffPost dataset
- Performance on real users, different news sources, or non-English content is **unknown and untested**
- The system should **not** be generalized to medical, legal, financial, or safety-critical recommendation without substantial additional validation, domain-specific datasets, and expert oversight

---

## 9. Commitment to Responsible Development

The research team is committed to the following principles throughout the project lifecycle:

| Principle | Implementation |
|---|---|
| **Privacy by Design** | No real user data collected or processed |
| **Fairness** | Per-category evaluation, oversampling of minority classes |
| **Transparency** | Open-source code, documented methodology, public dataset |
| **Accountability** | Named authors with institutional affiliation and contact information |
| **Non-maleficence** | Explicit rejection of manipulative use cases |
| **Beneficence** | Goal is improved information access, not engagement maximization |

---

## 10. References

- Pariser, E. (2011). *The Filter Bubble: What the Internet is Hiding from You.* Penguin Press.
- Misra, R. (2022). News Category Dataset. Kaggle. https://www.kaggle.com/datasets/rmisra/news-category-dataset
- Republic Act 10173 — Data Privacy Act of 2012 (Philippines). https://www.privacy.gov.ph
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. *WWW 2010*. https://doi.org/10.1145/1772690.1772758
- Wu, C. et al. (2019). Neural news recommendation with Multi-Head Self-Attention. *EMNLP 2019*. https://doi.org/10.18653/v1/d19-1671

---

*This ethics statement was prepared as part of the project submission requirements for Holy Angel University. It is intended to be a living document, updated as the system evolves.*
