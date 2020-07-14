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

* TSP has few constraints.
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

### 07/13/2020

#### 5. [Machine Learning for Combinatorial Optimization: a Methodological Tour d’Horizon](https://arxiv.org/pdf/1811.06128v2.pdf) - 2020

> This paper surveys the recent attempts, both from the machine learning and operations research communities, at leveraging machine learning to solve combinatorial optimization problems. Aiming at highlighting promising research directions in the use of ML within CO, instead of reporting on already mature algorithms.

* Motivations: Approximation and Discovery of new policies
  * With domain-knowledge, but want to alleviate the computational burden by approximating some of decisions with ML.
  * expert knowledge is not satisfactory, and wishes to find better ways of making decisions.


**Learning methods**

> In the case of using ML to approximate decisions: E.g.: Imitation Learning - blindly mimic the expert
>
> > ***Demonstration***
> >
> > 1. Baltean-Lugojan et al. (2018): cutting plane selection policy.
> >
> > 2. **Branching policies** in B&B trees of MILPs: The choice of variables to branch on can dramatically change the size of the B&B tree, hence the solving time.
> >
> >    *(i)* Inputs to the ML model are engineered as a vector of fixed length with static features descriptive of the instance, and dynamic features providing information about the state of the B&B process: Marcos Alvarez et al. (2014, 2016, 2017); Khalil et al. (2016)
> >
> >    *(ii)* Use a raw exhaustive representation(a bipartite graph) of the sub-problem associated with the current branching node as input to the ML model: Gasse et al. (2019)
> >
> >    Node selection is also a critical decision in MILP.
>
> In the case where one cares about discovering new policies: Reinforcement Learning
>
> > ***Experience***
> >
> > Some methods that were not presented as RL can also be cast in this MDP formulation, even if the optimization methods are not those of the RL community. **Automatically build specialized heuristics for different problems.** The heuristics are build by orchestrating a set of moves, or subroutines, from a pre-defined domain-specific collections. E.g. Karapetyan et al. (2017), Mascia et al. (2014).
>
> Demonstration and Experience are not mutually exclusive and most learning tasks can be tackled in both ways. 
>
> > E.g.: Selecting the branching variables in an MILP branch-and- bound tree could adopt anyone of the two prior strategies. Or an intermediary approach: Learn a model to dynamically switch among predefined policies during B&B based on the current state of the tree: Liberto et al. (2016)

**Algorithmic structure**

>**End to end learning:** Train the ML model to output solutions directly from the input instance.
>
><p align="center">
><img src="./img/5_1.png" alt="Editor" height="75"/></br>
></p>
>
>> Pointer Network/GNN + Supervised Learning/RL
>>
>> Emami and Ranka (2018) and Nowak et al. (2017): Directly approximating a double stochastic matrix in the output of the neural network to characterize the permutation.
>>
>> Larsen et al. (2018): Train a neural network to predict the solution of a stochastic load planning problem for which a deterministic MILP formulation exists.
>
>**Learning to configure algorithms:** ML can be applied to provide additional pieces of information to a CO algorithm. For example, ML can provide a parametrization of the algorithm.
>
><p align="center">
><img src="./img/5_2.png" alt="Editor" height="100"/></br>
></p>
>
>>  Kruber et al. (2017): Use machine learning on MILP instances to estimate beforehand whether or not applying a Dantzig-Wolf decomposition will be effective. 
>>
>> Bonami et al. (2018): Use machine learning to decide if linearizing the problem will solve faster.
>>
>> The heuristic building framework used in Karapetyan et al. (2017) and Mascia et al. (2014).
>
>**Machine learning alongside optimization algorithms:** build CO algorithms that repeatedly call an ML model throughout their execution.
>
><p align="center">
><img src="./img/5_3.png" alt="Editor" height="200"/></br>
></p>
>>This is clearly the context of the branch-and-bound tree for MILP.
>>
>>* Select the branching variable
>>* The use of primal heuristics
>>* Select promising cutting planes

**Learning objective**

>**Multi-instance formulation**
>$$
>\min_{a\in\mathcal{A}}\mathbb{E}_{i\sim p}m(i,a)
>$$
>**Surrogate objectives**
>
>The sparse reward setting is challenging for RL algorithms, and one might want to design a surrogate reward signal to encourage intermediate accomplishments.
>
>**On generalization**
>
>* new instances: Empirical probability distribution instead of true probability distribution.
>* other problem instances
>* unexpected sources of randomness
>* “structure” and “size”
>
>**Single instance learning**
>
>This is an edge scenario that can only be employed in the setting, where ML is embedded inside a CO algorithm; otherwise there would be only one training example! There is therefore no notion of generalization to other problem instances. Nonetheless, the model still needs to generalize to unseen states of the algorithm.
>
>Khalil et al. (2016): Learn an instance-specific branching policy. The policy is learned from strong-branching at the top of the B&B tree, but needs to generalize to the state of the algorithm at the bottom of the tree, where it is used.
>
>**Fine tuning and meta-learning**
>
>A compromise between instance-specific learning and learning a generic policy is what we typically have in multi-task learning. Start from a generic policy and then adapt it to the particular instance by a form of fine-tuning procedure.
>
>**Other metrics**
>
>Metrics provide us with information not about final performance, but about offline computation or the number of training examples required to obtain the desired policy. Useful to calibrate the effort in integrating ML into CO algorithms.

**Methodology**

> **Demonstration and Experience**
>
> Demonstration: the performance of the learned policy is bounded by the expert; learned policy may not generalize well to unseen examples and small variations of the task and may be unstable due to accumulation of errors.
>
> Experience: potentially outperform any expert; the learning process may get stuck around poor solutions if exploration is not sufficient or solutions which do not generalize well are found. It may not always be straightforward to define a reward signal. 
>
> It is a good idea to start learning from demonstrations by an expert, then refine the policy using experience and a reward signal. E.g. AlphaGo paper (Silver et al., 2016).
>
> **Partial observability**
>
> Markov property does not hold.
>
> A straightforward way to tackle the problem is to compress all previous observations using an RNN.
>
> **Exactness and approximation**
>
> In order to build exact algorithms with ML components, it is necessary to apply the ML where all possible decisions are valid.
>
> * End to end learning: the algorithm is learning a heuristic; no guarantee in terms of optimality and feasibility.
>
> * Learning to configure algorithms: applying ML to select or parametrize a CO algorithm will keep exactness if all possible choices that ML discriminate lead to complete algorithms.
>
> * Machine learning alongside optimization algorithms: all possible decisions must be valid.
>   * branching among fractional variables of the LP relaxation
>   * selecting the node to explore among open branching nodes (He et al., 2014)
>   * deciding on the frequency to run heuristics on the B&B nodes (Khalil et al., 2017b)
>   * selecting cutting planes among valid inequalities (Baltean-Lugojan et al., 2018)
>   * removing previous cutting planes if they are not original constraints or branching decision
>   * *Counter-example*: Hottung et al. (2017). In their B&B framework, bounding is performed by an approximate ML model that can overestimate lower bounds, resulting in invalid pruning. The resulting algorithm is therefore not an exact one.

**Challenges**

> **Feasibility**
>
> Finding feasible solutions is not an easy problem, even more challenging in ML, especially by using neural networks.
>
> End to end learning: The issue can be mitigated by using the heuristic within an exact optimization algorithm (such as branch and bound).
>
> **Modelling**
>
> The architectures used to learn good policies in combinatorial optimization might be very different from what is currently used with deep learning. RNN; GNN
>
> For a very general example, Selsam et al. (2018) represent a satisfiability problem using a bipartite graph on variables and clauses. This can generalize to MILPs, where the constraint matrix can be represented as the adjacency matrix of a bipartite graph on variables and constraints, as done in Gasse et al. (2019).
>
> **Scaling**
>
> A computational and generalization issue. All of the papers tackling TSP through ML and attempting to solve larger instances see degrading performance as size increases much beyond the sizes seen during training. 
>
> **Data generation**
>
> "*sampling from historical data* is appropriate when attempting to mimic a behavior reflected in such data".
>
> In other cases, i.e., when we are not targeting a specific application for which we would have historical data: 
>
> * Smith-Miles and Bowly (2015): Defining an instance feature space -> PCA -> using an evolutionary algorithm to drive the instance generation toward a pre-defined sub-space
> * Malitsky et al. (2016): generate problem instances from the same probability distribution using LNS.

### 07/20/2020

#### 6. A General Large Neighborhood Search Framework for Solving Integer Programs - 2020

> 

#### 7. Backdoors to Combinatorial Optimization: Feasibility and Optimality - 2009

> 

#### 8. Backdoors in the Context of Learning - 2009

> 