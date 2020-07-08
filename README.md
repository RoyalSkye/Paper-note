### 07/06/2020

#### 1. [Neural Combinatorial Optimization With Reinforcement Learning](https://arxiv.org/pdf/1611.09940) - 2016

> Pointer Network + Attention + RL to solve TSP problem.

Sequence to sequence problems like machine translation. But the vanilla sequence to sequence model (Encoder-decoder) has an issue: networks trained in this fashion cannot generalize to inputs with more than n cities.

<p align="center">
  <img src="./img/1_1.png" alt="Editor" width="600"/></br>
  Pointer Network (the output sequence length is determined by the source sequence.)
</p>
Supervised Learning: Generalization is poor. (1) the performance of the model is tied to the quality of the supervised labels, (2) getting high-quality labeled data is expensive and may be infeasible for new problem statements, (3) one cares more about finding a competitive solution more than replicating the results of another algorithm.

Actor: REINFORCE algorithm. b(s) estimates the expected tour length to reduce the variance of the gradients.
$$
\bigtriangledown_\theta J(\theta|s) = E_\pi[(L(\pi|s) - b(s))\bigtriangledown_\theta\log p_\theta(\pi|s)]
$$
Critic: Learn the expected tour length found by our current policy pθ given an input sequence s.
$$
L(\theta_v) = \frac{1}{B}\sum_{i=1}^B||b_{\theta_v}(s_i) - L(\pi_i|s_i)||_2^2
$$

<p align="center">
  <img src="./img/1_2.png" alt="Editor" width="600"/>
</p>


Problem and Challenge: 

* TSP has no constraints.
* The RNN of Pointer Network's encoder isn't necessary - Order doesn't matter.
* GENERALIZATION TO OTHER PROBLEMS:
  * by assigning zero probabilities to branches that are easily identifiable as infeasible.
  * consists in augmenting the objective function with a term that penalizes solutions for violating the problem’s constraints.

#### 2. [Reinforcement Learning for Solving the Vehicle Routing Problem](https://papers.nips.cc/paper/8190-reinforcement-learning-for-solving-the-vehicle-routing-problem.pdf) - 2018

> One vehicle with a limited capacity is responsible for delivering items to many geographically distributed customers with finite demands.

* One major issue that complicates the direct use of [1] for the VRP is that it assumes the system is static over time. In the VRP, the demands change over time.
  $$
  x_t^i = (s_i, d_t^i)
  $$
  For instance, in the VRP, $$x_t^i$$ gives a snapshot of the customer i, where $$s_i$$ corresponds to the 2-dimensional coordinate of customer i’s location and $$d_t^i$$ is its demand at time t. 

* RNN encoder adds extra complication to the encoder but is actually not necessary.

<p align="center">
  <img src="./img/2_1.png" alt="Editor" width="400"/>
  <img src="./img/2_2.png" alt="Editor" width="400"/></br>
</p>

> Experiment:
>
> The node locations and demands are randomly generated from a fixed distribution. 
>
> * Location: [0, 1] * [0, 1]
> * Demand: [1, 2, ..., 9] or continuous distribution.
>
> After visiting customer node i, the demands and vehicle load are updated.
>
> Using negative total vehicle distance as the reward signal.

Satisfying capacity constraints by **masking scheme**.

*(i)* nodes with zero demand are not allowed to be visited; *(ii)* all **customer** nodes will be masked if the vehicle’s remaining load is exactly 0; and *(iii)* the customers whose demands are greater than the current vehicle load are masked.

The solution can allocate the demands of a given customer into multiple routes by relaxing *(iii)*.

> **Dynamically changing VRPs.**
>
> For example, in the VRP, the remaining customer demands change over time as the vehicle visits the customer nodes; or we might consider a variant in which new customers arrive or adjust their demand values over time, independent of the vehicle decisions.

#### 3. [Learning to Branch in Mixed Integer Programming](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12514/11657) - 2016

> Branch-and-Bound: Two of the main decisions to be made during the algorithm are node selection and variable selection.

Traditional branching strategies fall into two main classes: 

* Strong Branching (SB) approaches exhaustively test variables at each node, and choose the best one with respect to closing the gap between the best bound and the current best feasible solution value. 
  * fewer search tree nodes
  * more computation time
* Pseudocost (PC) branching strategies are **engineered to imitate SB** using a fraction of the computational effort, typically achieving a good trade-off between number of nodes and total time to solve a MIP. 
  * based on human intuition and extensive engineering
  * requiring significant manual tuning

Two of the main decisions to be made during B&B:

* Node selection: Select an active node N. Following that, the LP relaxation at N is solved. Based on ub/lb to find the nodes needed to be pruned.
* Variable selection: Given a node N whose LP solution X is not integer-feasible. Let C is the set of branching candidates whose value is not integer in X. Calculate the score *(SB/PC)* for C and choose a variable based on it.

<p align="center">
  <img src="./img/3_1.png" alt="Editor" width="500"/></br>
  Branch-and-Bound
</p>
**Learn branching strategies directly from data. (data-driven, on-the-fly design of variable selection strategies)**

> A machine learning (ML) framework for **variable selection** in MIP. It observes the decisions made by Strong Branching (SB), a time-consuming strategy that produces small search trees, collecting features that characterize the candidate branching variables at each node of the tree. 
>
> Learn a function of the variables’ features that will rank them in a similar way, without the need for the expensive SB computations.
>
> *(i)* can be applied to instances on-the-fly.
>
> *(ii)* consists of solving a ranking problem, as opposed to regression or classification.

Overview:

> *(i)* Data collection: for a limited number of nodes i, SB is used as a branching strategy. At each node, the computed SB scores are used to assign labels to the candidate variables; and corresponding variable features are also extracted.
>
> *(ii)* Learning: the dataset is fed into a learning-to-rank algorithm that outputs a vector of weights for the features.
>
> *(iii)* ML-Based Branching: Learned weight vector is used to score variables instead SB, which is computationally expensive.

Learning-to-rank: $$l(.)$$ is based on *pairwise loss*, a common IR measure.

$$
w^*=\mathop{\arg\min}_{w}\sum_N\ell(y^i - \hat{y}^i) + \lambda||w||_2^2
$$

<p align="center">
  <img src="./img/3_2.png" alt="Editor" width="700"/>
</p>
> Future:
>
> * may also used for other components of the MIP solver such as cutting planes and node selection.
> * RL instead of the batch supervised ranking approach.

#### 4. [A General Large Neighborhood Search Framework for Solving Integer Programs](https://arxiv.org/pdf/2004.00422) - 2020

> LNS + Learning a Decomposition
>
> Design abstractions of large-scale combinatorial optimization problems that can leverage existing state-of-the-art solvers in general purpose ways. A general framework that avoids requiring domain-specific structures/knowledge.

* Decomposition-based LNS for Integer Programs

  > Select a subset and use an existing solver to optimize the variables in that subset while holding all other variables fixed.
  >
  > Random decomposition empirically already delivers very strong performance.

<p align="center">
  <img src="./img/4_1.png" alt="Editor" width="400"/></br>
</p>
* Learning a Decomposition

  > *(i)* Reinforcement Learning: 
  >
  > State is a vector representing an assignment for variables in X, i.e., it's an incumbent solution.
  >
  > Action is a decomposition of X.
  >
  > Reward $$r(s,a) = J(s) - J(s')$$, which is the difference of obj between old and new solution.
  >
  > *(ii)* Imitation Learning:
  >
  > Generate a collection of good decompositions: by sampling random decompositions and take the ones resulting in best objectives as demonstrations. See Alg 2.
  >
  > Then apply behavior cloning, which treats policy learning as a supervised learning problem.

<p align="center">
  <img src="./img/4_2.png" alt="Editor" width="400"/></br>
</p>



