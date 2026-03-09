Understanding Gradient Descent: Batch, Stochastic, and Normal Equation
After working through CS229 Lecture 2, I wanted to share my understanding of gradient descent and the different approaches to optimizing linear regression. This lecture covered three methods to find optimal parameters: batch gradient descent, stochastic gradient descent, and the normal equation.
The Problem: Finding the Best Fit
In linear regression, we're trying to find parameters θ that minimize our cost function J(θ). Our hypothesis function is:
h(θ) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
The cost function we use is Mean Squared Error (MSE):
J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
Why MSE? Because it's quadratic, which creates a bowl-shaped graph with a single global minimum - no local minima to worry about. This makes it a convex optimization problem.
Why the 1/2? It's mathematical convenience. When we take the derivative, the power of 2 brings down a 2 that cancels with 1/2, making our gradient formula cleaner. The 1/m gives us the mean (average error across all examples).
The Gradient: Which Direction to Move?
The gradient ∇J(θ) points in the direction of steepest ascent - the direction where the cost function increases fastest. But we want to minimize the cost, not maximize it. That's why the update rule subtracts the gradient:
θ := θ - α∇J(θ)
Here, α (alpha) is the learning rate - it controls how big our steps are.
Learning rate intuition: Imagine cycling toward a dip in the road. If you pedal too slowly (small learning rate), you'll reach the bottom eventually, but it takes forever. If you pedal too fast (large learning rate), you'll gain so much speed that you fly right over the dip and land on the other side - completely missing the bottom. You might then pedal back, overshoot again, and keep jumping back and forth across the dip without ever settling at the bottom.
A typical starting value for α is 0.01, but it depends on the specific problem.
Batch Gradient Descent: The Careful Approach
Formula:
θⱼ := θⱼ - α(1/m)Σᵢ₌₁ᵐ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
Batch gradient descent uses ALL m training examples to compute the gradient at each step. This means it takes the steepest path downhill and converges smoothly to the global minimum.
The drawback? It's computationally expensive. For every single parameter update, you must process all m examples. If you have a million training examples, that's a million computations per step. For large datasets, this becomes painfully slow.
Stochastic Gradient Descent: The Fast & Noisy Approach
Formula:
θⱼ := θⱼ - α(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
Stochastic gradient descent (SGD) updates parameters using just ONE example at a time. Instead of waiting to see all examples before taking a step, it updates immediately after each example.
Why is the path noisy? Each training example pulls the gradient in a different direction. Some examples have small errors (small updates), others have large errors (large updates). The result is a zigzag path that wobbles toward the minimum instead of following the steepest descent.
Here's an analogy: Batch gradient descent is like a sober person who checks all available sources - Google Maps, asks for directions, studies the route - and then takes the optimal path to arrive precisely at the destination. Stochastic gradient descent is like a drunk person who has a rough idea where to go and gets there faster (doesn't stop to check everything), but takes a wobbly path and keeps wandering around the destination instead of stopping precisely there.
Key behavior: SGD oscillates AROUND the minimum. It crosses the minimum point repeatedly but never truly converges to it. Why? Because it's always looking at the next random example, which pulls it in a new direction. The size of these oscillations depends on the learning rate - higher α means larger oscillations with more variance, lower α means tighter wobbling around the minimum.
The advantages: SGD is much faster per iteration (one example instead of all examples) and uses far less memory (loads one example at a time instead of the entire dataset). For most practical purposes, getting close to the minimum is good enough - you don't need the exact optimal parameters.
Normal Equation: The Direct Solution
Formula:
θ = (X^T X)^(-1) X^T y
The normal equation solves for θ directly without any iterations. How? By setting the gradient equal to zero: ∇J(θ) = 0.
Why does gradient = 0 give us the minimum? When the derivative equals zero, the slope is zero. For our quadratic cost function (bowl-shaped graph), there's only one point with zero slope: the bottom of the bowl, which is the global minimum.
When it fails: The normal equation requires computing (X^T X)^(-1). If X^T X is not invertible (singular matrix), this fails. This happens when features are linearly dependent - for example, if you have both height in centimeters and height in inches as features (one is redundant). The solution is to remove redundant features.
Computational cost: Matrix inversion is O(n³) where n is the number of features. For small n (less than 10,000 features), this is fast. For large n, it becomes slow - slower than gradient descent.
Which Method Should You Use?
Use Normal Equation when:

Small number of features (n < 10,000)
You want the exact solution
No iteration needed

Use Batch Gradient Descent when:

Medium-sized datasets
You want precise convergence
You have computational resources

Use Stochastic Gradient Descent when:

Large datasets (millions of examples)
You need fast iterations
Approximate solution is acceptable

Memory comparison: Batch gradient descent must load all m examples into memory simultaneously. Stochastic loads only one example at a time. For a dataset with 1 million examples, batch GD needs roughly 1000x more memory than stochastic GD.
Speed comparison: Consider 1 million training examples. If batch GD needs 100 steps to converge, that's 100 million total computations (100 steps × 1 million examples per step). If stochastic GD needs 10,000 steps to converge, that's only 10,000 computations (10,000 steps × 1 example per step). Stochastic is 1000x faster in this case.
Real-world note: In practice, most modern ML uses mini-batch gradient descent - batches of 32, 64, 128, or 256 examples. This balances the stability of batch GD with the speed of stochastic GD.
Conclusion
Understanding these three approaches gives you the tools to choose the right optimization method for your problem. Batch gradient descent is precise but slow. Stochastic gradient descent is fast but noisy. The normal equation is direct but computationally expensive for large feature sets. Each has its place, and knowing when to use which is key to practical machine learning.
Test Your Understanding
If you've understood this lecture, you should be able to answer these questions:
On Batch Gradient Descent:
1. Given X = [[1, 2], [1, 3]], y = [5, 7], θ = [0, 0], and α = 0.1, what is θ after one iteration of batch gradient descent?
2. Why does the cost function J(θ) never increase during batch gradient descent (assuming correct learning rate)?
3. What is the time complexity per iteration for batch GD with m examples and n features?
On Stochastic Gradient Descent:
4. Can SGD ever reach the exact global minimum? If yes, will it stay there?
5. Why does SGD use less memory than batch GD? (Hint: it's NOT about number of iterations)
6. For a dataset with 1 million examples, why is SGD faster than batch GD even though it might need more iterations?
7. What happens to oscillation size if you increase the learning rate in SGD?
8. Is the "noise" in SGD always bad? Why or why not?
On Normal Equation:
9. What does it mean mathematically when we say ∇J(θ) = 0?
10. Under what conditions does (X^T X)^(-1) not exist?
11. If you have 100,000 features, would you use normal equation or gradient descent? Why?
Comparison:
12. You have 1 million training examples and 50 features. Rank these methods from fastest to slowest: Batch GD, Stochastic GD, Normal Equation.
