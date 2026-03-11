# Daily Learning Log

## 2026-03-12 (Thursday)

**Time Spent:** 3 hours

**What I Did:**
- Completed majority of CS229 Lecture 3 (Logistic Regression section, 63/77 min)
- Deep dive on logistic regression mechanics: sigmoid function, probability interpretation, likelihood derivation, gradient descent vs gradient ascent
- Understood key concepts: why sigmoid gives (0,1) range, why multiply probabilities (IID assumption), gradient intuition (h-y = "how wrong we are" vs y-h = "gap to close")
- Clarified confusions: sigmoid range, combining probability equations, why maximize likelihood = minimize cost

**Output:**
- Solid conceptual understanding of logistic regression
- Can explain: sigmoid squashing, probability formulation, likelihood maximization, gradient update rules
- Clear intuition for gradient descent (subtract error) vs gradient ascent (add gap)

**Notes:**
- Logistic regression initially confusing (many concepts at once) but intuition clicked after working through step-by-step
- Gradient descent vs ascent perspective: same algorithm, different framing (minimize cost vs maximize likelihood)

---

## 2026-03-11 (Wednesday)

**Time Spent:** 2.5 hours

**What I Did:**
- Watched first 42 min of CS229 Lecture 3 (locally weighted regression)
- Deep dive on probabilistic interpretation of MSE
- Understanding test on locally weighted regression (11/12 = 91.7%)

**Output:**
- Solid understanding of locally weighted regression (non-parametric, computational cost, bandwidth parameter)
- Conceptual grasp of probabilistic interpretation (MSE from Gaussian likelihood)

**Notes:**
- Struggled with probabilistic interpretation 
- Understood Locally Weighted Regression
---

## 2026-03-10 (Tuesday)

**Time Spent:** 3 hours

**What I Did:**
- Completed CS229 Lecture 2 understanding test (8/10 score)
- Corrected misunderstandings: memory vs iterations in SGD, noise not always bad (in case of linear regression bad-neutral), SGD oscillation behavior
- Wrote first blog post: "Understanding Gradient Descent: Batch, Stochastic, and Normal Equation"
- Published blog to GitHub repo

**Output:**
- Blog post: `blog/cs229-lecture2-gradient-descent.md`
- Understanding solidified through writing.

**Notes:**
- Writing blog helped clarify concepts I thought I understood but didn't fully
- Cycling analogy worked well for explaining learning rate
---

## 2026-03-09 (Monday)
**Time Spent:** 8 hours

**What I Did:**
- Completed Python/NumPy revision (W3Schools + NumPy docs)
- Implemented Linear Regression - Normal Equation (Deep-ML ✓)
- Implemented Linear Regression - Gradient Descent (Deep-ML ✓)
- Started CS229 Lecture 1 (ML types: supervised, unsupervised, RL)
- Watched CS229 Lecture 2 (first 45 min: linear regression, cost function, batch GD)

**Output:**
- 2 working implementations (normal equation + gradient descent)
- Understanding test: 8/10
- GitHub repo created

**Key Learnings:**
- Cost function J(θ) measures average squared error
- Gradient descent: iteratively minimize J(θ) by moving opposite to gradient
- Batch GD uses all m examples per update
