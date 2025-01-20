# Nature inspired computation

#### Evolutionary and natural computation
* evolution as an algorithm &rarr; progress, adaptation, learning, survival of the fittest

Template of evolutionary program:
```python
generate a population of agents (objects, data structures)
do {
    compute fitness (quality) of the agents
    select candidates for the reproduction using fitness
    create new agents by combining the candidates
    replace old agents with new ones
} while (not satisfied)
```

#### Genetic algorithms
* Holland, 1970's, based on Darwinian evolution
* genes represented as bit/numeric vectors/strings...
* applications: optimization, scheduling, bioinformatics, ML, planning...
* **crossover**: 
    + single point, multipoint
    + linear (linear combination of 2 individuals $\alpha x + (1-\alpha)y$)
    + permuation and ordered crossover (keep part of sequences to not get invalid solutions)
    + adaptive (different evolution phases and crossover templates)
* **mutation**:
    + sffing nre information
    + single point/multipoint
    + Lamarckianism (organism can pass on characteristics that it has acquired through use/disuse during its lifetime to its offsprint - giraffe)
    + Gaussian (selects a position in the vecot of floats and mutates ut by adding Gaussian error)

#### Selection - who will reproduce?
+ **proportional**
+ **rank proportional**
+ **tournament:** randomly sample $t$ sgents from population, select the best with probability $p$, second best with $p(1-p)$...
+ **single tournament:** randomnly split population into small groups, apply crossover to 2 best agents from each group, their offspring replace two worst agents from group
+ **stohastic universal sampling**

Replacement - all, according to fitness (roulette, tournament, random), elitism (portion of the best), local elitism (children replace parents if they are better)
Stopping criteria - number of generations, track progress, avaliability of computational resources...
Multiobjective optimization - fitness function with several objectives, $min F(x) = min (f_1(x),...,f_n(x))$

#### Strenghts and weaknesses
+ robust, adaptable, general, requires weak knowledge of the problem, several alternative solutions, parallelization, faster and less memory than search
+ suboptimal solutions, possibly many parameters, can be computationally expensive
+ **no-free-lunch theorem** (an algorithm that performs exceptionally well on one class of problems must compensate by performing worse on another class of problems)

#### Neuroevolution
+ evolving neuros/topologies
+ evolving weights instead of backpropagation/gradient descent


# Statistical Predictive Modeling

#### Learning
+ act of acquiring new or modifying and reinforcing existing knowledge, behaviours, skills, values, preferences
+ synthesizing different types of information
+ statistical learning &rarr; finding a predictive function based on data
+ basic task in ML &rarr; learning from past examples with known outcomes

#### Notation
+ the variable we wish to predict/target: $Y$
+ input variables/attributes, features: $X_i$
+ input vectors form a matrix: $X = \begin{pmatrix} x_{11} & x_{12} & \cdots & x_{1p} \\ x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix} $
+ **model: $Y = f(X) + \epsilon$**
+ $Y_i = f(X_i)+\epsilon_i$

#### Goals of learning
+ **prediction:**
    + if we make a good estimate for $f$ and we have small variance of $\epsilon$, we can make accurate predictions for $Y_i$ based on new $x_i$
+ **inference:**
    + relationship between $Y$ and $X_i$

#### Statistical learning methods

**Parametric methods:**
+ reduce the problem of estimating $f$ to estimating a set of parameters in 2 steps
+ **step 1** &rarr; come up with a model of $f$ (usually linear): $f(X_i) = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_p X_{ip}$
+ **step 2** &rarr; use training data to fit the model and estimate unknown parameters $\beta_i$

**Non-parametric methods:**
+ no assumptions about $f$
+ advantage &rarr; accurately fit a wider range of possible shapes of $f$
+ disatvantage &rarr; need a large amount of observations

**Trade-off between prediction accuracy and model interpretability:**
+ simple methods produce models that are easier to interpret
+ it's possible to get more accurate predictions with simpler models (hard models are harder to git)

#### Dividing learning problems

+ **Supervised learning:**
    + both the predictions ($X_i$) and the response ($Y_i$) are observed
+ **Unsupervised learning:**
    + only observe the predictions ($X_i$) 
    + use the predictions to guess the response and then build a model *(clustering)*
+ **Semi-supervised learning:**
    + small sample of labelled instances are observed, a large sample of unlabeled instances
    + initial supervised model used to label unlabeled instances
    + add the ost reliable predictions to the training set for next iteration
+ **Self-supervised learning:**
    + learns from unlabeled data
    + labels obtained from related properties of the data
    + predicts unobserved/hidden property of the input
+ **Weakly-supervised data:**
    + noisy sources used to provide supervision signal for labeling large amounts of training data for supervised learning

#### Regression VS classification
We can split **supervised learning problems** into regression and categorical problems. In **regression** problems $Y$ is continuous and in **classification** problems, $Y$ is categorical. Some methods

#### Data mining
Association and correlation analysis:
+ frequent patterens
+ association, correlation VS causality

Outlier analysis:
+ **outlier** = data object that does not comply with the general behaviour of the data
+ noise or exception?

#### Criteria of success for ML
The goal of classification is minimizing the test error. Many algorithms solve optimization problems - minimizing the error.

Criteria of success for ML:
+ **regression:**
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - f'(x_i))^2$$
+ **classification:**
$$\text{CA} = \frac{1}{n} \sum_{i=1}^n I(y_i = y'_i)$$

#### No-Free-Lunch theorem
If no information about the target function $f$ is provided:
+ generally no classifier is better than some other
+ generally no classifier is better than random