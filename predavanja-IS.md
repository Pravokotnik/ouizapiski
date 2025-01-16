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