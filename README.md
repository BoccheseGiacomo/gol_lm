# Gol-LM
A Language Model based on Game Of Life Cellular Automata with potential meta learning capabilities.

*Gol-LM © 2024 by Giacomo Bocchese is licensed under CC BY-NC-SA 4.0 with additional conditions specified in the LICENSE file.*

## Abstract
Gol-LM is an experimental research project focused on developing an adaptive language model using the principles of the Game of Life (GoL) cellular automaton. By harnessing the Turing completeness of GoL (finite memory approximation), Gol-LM enables the formation and evolution of sophisticated internal algorithms within a two-dimensional grid. The model employs genetic algorithms for evolutionary optimization, positioning Gol-LM as a robust and flexible language model capable of complex computational tasks.

Distinctive to Gol-LM is its ability to achieve spontaneous meta-learning, facilitating the emergence of self-reinforcement learning and dynamic memory organization. This capability allows the model to autonomously evolve internal optimization strategies, adapting its behavior based on external rewards. This mirrors natural processes where complexity and adaptability arise from non-linear, interactive systems that evolve towards optimal solutions.

As a language model, Gol-LM translates input sequences into coherent language predictions through iterative evolution and adaptation. The model's learning process is guided by external feedback, which informs its internal state adjustments and enhances its predictive accuracy over time.

It is important to note that Gol-LM is in a highly experimental stage. Many theoretical aspects are derived from rigorous deduction and intuition based on known principles in machine learning and emergent systems, but may not be proved enough. The forthcoming phases of the project are exploratory, aiming to systematically uncover the model's capabilities and constraints. These steps are essential for transitioning from theoretical constructs to empirical results, advancing our understanding of cellular automata-based language modeling. The training part is not implemented yet.

This work is inspired from my previous project: the [Convolutional Turing machine](https://github.com/BoccheseGiacomo/ConvolutionalTuringMachine). The key difference between the two is that this uses discrete states (Conway's GOL), while the other uses convolution-based cellular automata (similar to Lenia cellular automata). Furthermore, this was developed with a specific focus on language modeling.

<div>
  <img src="https://github.com/user-attachments/assets/f1773f5a-1ddb-4e8c-9bde-c93dec488601" alt="Gol-LM Simulation" width="auto" height="auto">
  <p><em>Figure 1: A showing the inference process of a random, non trained, Gol-LM.</em></p>
  <p><em>State space dimension: 61x50 = 3050 cells</em></p>
  <p><em>Vocabulary dimension: 11 ; Symbols : ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '|']</em></p>
  <br>
</div>

## Citation
```bibtex
@misc{gol_lm2024,
  author = {Giacomo Bocchese},
  title = {Gol-LM: A Language Model based on Game Of Life Cellular Automata with potential meta learning capabilities},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BoccheseGiacomo/gol_lm}}
}
```

## Project Vision and Objectives

Gol-LM is envisioned as a language model with an inherent capacity for learning and adaptability, built upon the principles of the Game of Life (GoL) cellular automaton. The central goal of Gol-LM is to demonstrate the emergence of sophisticated language modeling behaviors and the ability to develop internal algorithms, given sufficient two-dimensional space and computation time. The project is anchored in several key objectives:

**Complex Behavior Simulation**: The GoL rules serve as a powerful mechanism to simulate complex behaviors within Gol-LM. By evolving patterns that mimic the dynamics observed in nonlinear systems, Gol-LM can model intricate processes, potentially emulating the computational structures required for advanced language tasks. This capability opens the door to representing and manipulating the spatial configurations of information, akin to modeling neural networks or other complex systems.

**Halting and Flow Control**: Introducing a halting mechanism in Gol-LM is crucial for expanding its computational repertoire, particularly for implementing flow control in language processing tasks. This mechanism allows for controlled looping and iterative operations, broadening the range of computable functions. By managing computation length and iterative processes, Gol-LM enhances its ability to represent diverse language functions through autoregressivity. This aspect is integral to achieving Turing completeness, where the model can potentially simulate any computation given unbounded state space and time. Experimentation with different state and kernel dimensions, as well as the potential incorporation of non-linear operations, will help refine this capability.

**Meta-Learning**: Gol-LM is designed to autonomously develop internal reinforcement learning algorithms, driven by external rewards. These internal algorithms enable the model to learn and optimize its language processing capabilities more efficiently than relying solely on external optimization methods such as genetic algorithms. Through meta-learning, Gol-LM can adaptively improve its performance and generalize across various language tasks.

**Self-Reinforcement Learning**: Gol-LM hypothesizes the emergence of "Self-Reinforcement Learning", where the model evolves internal reward mechanisms to guide its behavior towards optimal language modeling. This involves a form of cognitive simulation or 'imagination', where Gol-LM internally generates scenarios, hypothesizes outcomes, and creates synthetic data and rewards. This self-simulation allows the model to strategize and evolve, mirroring the human ability to mentally forecast and plan before actual execution. By leveraging self-generated data, Gol-LM aims to achieve higher data efficiency and better generalization.

**Enhanced Reasoning Capabilities**: Gol-LM aims to exhibit superior reasoning capabilities compared to neural network-based algorithms. By incorporating flow control and halting mechanisms, Gol-LM can learn and execute complex algorithms rather than relying on approximation or interpolation, which often leads to raw memorization. This approach is expected to enhance the model's grokking and generalization capabilities, enabling it to better understand and process language tasks.

**Evolutionary Training**: The training process of Gol-LM relies on genetic algorithms (or other black box optimization algorithms), emphasizing the exploration of diverse initial states and evolutionary strategies. This approach aligns with the non-differentiable nature of GoL and facilitates the iterative improvement of internal algorithms. Through the selection of the best-performing states and the application of genetic operations, Gol-LM evolves to enhance its language modeling capabilities over successive generations, and may learn to exploit reward information in order to build internal emergent optimizers. This training part has been extensively planned but it's not implemented yet in the code.

**Not only language modeling**: The model is initially developed for language modeling tasks, but with minimal modifications it can be adapted to other settings, like driving an agent in RL settings.

Gol-LM represents a cutting-edge exploration into the potential of cellular automata for advanced language modeling, combining principles of Turing completeness, meta-learning, and evolutionary optimization to push the boundaries of adaptive computational systems.


## Internal Mechanics Description

### Initial Setup and Configuration

Gol-LM operates on a two-dimensional grid, with each cell representing its state within the Game of Life (GoL) cellular automaton. The grid is divided into specific regions:

![gollmpreview](https://github.com/user-attachments/assets/339917d0-55de-4d41-9a30-a993c80bbdf3)

- **Input Cells:** On the left side, activated by one-hot encoding of the input token. (White cells on the left).
  - There is an input cell for every vocabulary symbol. In this case the numbers 0-9 for arithmetic settings + the EOS token "|".
- **Output Cells:** On the right side, where alive cells increase the output count when touching the boundary, building a probability distribution over the vocabulary. (White cells on the right).
- **Reward Cells:** Distributed within horizontal sides to receive external rewards for guiding learning. (Not shown in this chart, but they are in the upper-center and lower-center sides).
- **Special output cell EOS:** On the output side (but also on the input side for symmetry reasons), there is a special token "|". It represents the End Of Sentence condition. When it is sampled as an output token, the generation of new tokens stops.

*Note: in reality, since it's a better approach for encoding information into GOL, we don't activate a single cell per one-hot input, but a range of 3 cells around the input cell. This way input can propagate more effectively into the state space*
  
### State Evolution

Gol-LM uses GoL rules to evolve the grid's state. The interaction between cells is local, driven by these rules, allowing complex behaviors and computations to emerge. 
It's proven that given a certain set of rules and an infinitely extended state space, GOL is turing complete. In this case we have a finite state space, which implies a limited maximum complexity of the algorithm we can represent (equivalence with a finite state machine). By extending the state space, maximum complexity can be increased.
However, since Gol-LM uses iterative computation with halting condition, the parameter dimension (in this case state space) needed to represent a complex algorithm is greatly reduced if compared to a neural network (that works in a single iterative setting for next token prediction).[This is highly likely but needs to be proved with further research].
An iterative algorithm can generalize better using a lower count of parameters since it reaches greater expressivity if compared to non-iterative algorithms, that need to map from *x* to *y* in a single iteration.
Iteration allows for better reasoning and logic capabilities of language modeling.
Gol-LM dynamically determines halting based on output accumulation, ensuring efficient computation tailored to input and task complexity. This iterative process allows Gol-LM to represent complex internal algorithms for superior language modeling.

### Inference and Processing

1. **Receiving Input token:** Input tokens activate the corresponding input cell.
2. **State Evolution:** GoL rules iteratively evolve the grid state.
3. **Output Accumulation:** Alive cells touching the right boundary of the state space increase the output accumulation count, depending on the height coordinate, forming a probability distribution.
5. **Halting Condition:** Iterations stop when the total sum of output accumulation exceeds a threshold.
6. **Output Retrieval and Sampling:** The output distribution is used to sample the next token via temperature-based stochastic sampling.
7. **Reward Integration:** Optionally, a reward feedback can be given to the state space, by setting some specific cells in the state space as "alive". In this case an additional iterative loop until halting is added for meta-learning computation. This meta-learning functionality works only if previously trained (see training below), if not, it will not add any improvement to the model performance.
8. **Repeat**: The model continues to generate tokens until the "End Of Sentence token", represented via a "|" in this case. In this way an entire sentence is generated.

*Note: there are two nested iterative processes; the internal loop governs the process of generating a single token, and continues until a halting condition determined by output accumulation going over a threshold. The external loop governs the generation of an entire sentence token by token, and halts when the end of sentence token is generated.*
*Furthermore, the memorization of the sentence is done inside the state space. Given the token **x_t** , we generate the token **x_t+1**, like in RNNs and differently from the transformers, where the entire sequence is given as input at each time step. This may arise long term memory issues, but the genetic algorithm training should allow for the development of internal mechanism to balance short and long term memory in order to minimize loss. Obiously, the larger is the state space the larger is the amount of storable memory*

## Challenges and Training

Training Gol-LM presents unique challenges due to its reliance on the Game of Life (GoL) cellular automaton and its non-differentiable nature. Unlike conventional optimization methods used in Artificial Neural Networks (ANNs), such as gradient descent, Gol-LM employs a more exploratory approach to training:

**Genetic Algorithms for Exploration:** Gol-LM training primarily relies on genetic algorithms (GAs), which are well-suited for exploring a diverse range of policies and behaviors. GAs facilitate the evolution of the model by selecting and propagating the most successful initial states and configurations over successive generations. This method encourages experimentation and discovery of optimal strategies for language modeling and theoretically allows for the emergence of internal meta-learning.

**Complex Training Dynamics:** Leveraging the theoretical capabilities of GoL within Gol-LM is an ambitious task. The training process involves enabling the model to propagate information effectively, construct intricate internal algorithms, and develop reinforcement learning mechanisms that improve its performance over time. The model's ability to achieve sophisticated language understanding and generation through iterative state evolution requires careful tuning and exploration.

**Alternatives:** While genetic algorithms are the primary focus due to their versatility and minimal assumptions about the data and processes, other zeroth-order optimization techniques can also be considered, like Simulated Annealing, Particle Swarm Optimization, Bayesian Optimization or some RL techniques.

## Training Phase of Gol-LM with Genetic Algorithms

### Components and Initialization

In the training phase of Gol-LM, the primary component subject to training is:

- **Initial State of the Grid:** This includes the entire grid or a subset of it, with some parts potentially fixed to a specific state to ensure stability and functionality. The GoL rules are typically kept fixed, given their Turing completeness, allowing the model to represent any computable function by modifying the initial state.

The training process begins with the random initialization of these initial states across a population of \( n \) Gol-LM models. Each model in the population is uniquely characterized by its own initial state configuration.

### Simulation and Selection

1. **Running Simulations:** For each model in the population, a simulation is run where the model processes given input tokens and generates output tokens. During this phase, the model evolves through the GoL rules, and the output is evaluated based on its alignment with the expected next token. Random mutations can also be applied in runtime to introduce variability and encourage exploration of the solution space.

2. **Evaluation and Reward Calculation:** The performance of each model is assessed based on how accurately the generated outputs match the expected next tokens. This can be quantified using metrics such as prediction accuracy or other suitable loss functions. In future versions, human feedback can also be integrated to refine the reward calculation.

3. **Selection of the Best Models:** Post-simulation, models are evaluated based on their performance. The best-performing models, those that most accurately predicted the next tokens and utilized the reward effectively, are selected for the next stage.

4. **Reproduction with Genetic Operators:** The selected models undergo reproduction, where genetic operations like mutation and crossover are applied. This step generates a new population of models, inheriting and varying traits from the successful models of the previous generation. Random mutations are also introduced to maintain diversity and prevent premature convergence.

5. **Iterative Process:** This process of simulation, evaluation, selection, and reproduction is repeated over multiple generations. With each iteration, the models are expected to progressively improve in their language modeling capabilities, developing more sophisticated and effective internal algorithms.

### Continual Learning and Stability

Even when the model is fully trained, Gol-LM is designed to continually apply multiple copies that evolve using genetic algorithms. This continual learning approach ensures that the model can adapt to new data and evolving language patterns over time. By maintaining a dynamic population of models, Gol-LM can mitigate potential instabilities in the state space, ensuring stability and convergence. This ongoing evolutionary process helps prevent divergence and maintains the robustness of the algorithm.

### Emergence of Meta-Learning

The emergence of meta-learning within Gol-LM is anticipated due to the system's inherent flexibility and capacity to model a wide range of algorithms, including internal optimizers. Theoretically, if a Gol-LM instance, by chance, develops the ability to internally update its state in response to external rewards, it would demonstrate faster and more efficient learning compared to other instances. Such a model would have a higher likelihood of being selected in the genetic algorithm (GA) process, thereby propagating its traits to subsequent generations.

Over time, this evolutionary pressure leads to the dominance of models capable of such internal learning optimizations, making meta-learning an emergent standard within the population. This process hinges on the principle that models which can internally incorporate reward information to refine their state will outperform and outlast those that do not. Consequently, models with meta-learning capabilities will naturally emerge and become prevalent in the Gol-LM population.

The same evolutionary principles apply to self-reinforcement learning. Models that, by chance, develop the ability to engage in self-reinforcement learning—where internal reward mechanisms guide behavior towards optimal solutions—will have a selective advantage. These models simulate scenarios, hypothesize outcomes, and generate synthetic data and rewards internally, allowing them to strategize and evolve towards optimal behaviors more effectively.

## Current State and Future Directions

### Current Progress

As of now, the Gol-LM project is in its early developmental phase. The foundational model has been established, but it has yet to be applied or tested in practical scenarios. Key aspects like training methodologies and their effectiveness in achieving desired behaviors are still theoretical and await empirical validation. To date, no concrete results have been achieved; the project remains in the realm of setup and initial exploration.

### Future Steps

Looking ahead, the project has several critical milestones to achieve:

**Implementing a Genetic Algorithm:** The initial phase of training will involve the implementation of a genetic algorithm. This step is fundamental for basic training and setting the groundwork for more advanced learning processes.

**Task Learning Without Reward:** An early experiment will involve attempting to train Gol-LM to learn a specific task using the genetic algorithm alone, without any external reward mechanism. This test aims to explore Gol-LM's capabilities in a more constrained learning environment.

**Extensive Training with Rewards:** The subsequent phase will focus on extensively training Gol-LM across a variety of tasks while incorporating external rewards. The goal here is to observe and measure the emergence of meta-learning. This stage is crucial for understanding whether and how Gol-LM develops its internal learning strategies and adapts to diverse challenges.

**Transitioning to 3-Dimensional Cellular Automata:** To enhance the complexity of the behaviors, we plan to transform Gol-LM to a 3D cellular automaton system. By adding an additional dimension, we aim to increase the model's capacity for storing and processing information, potentially leading to more sophisticated and stable behavior. This transition is expected to facilitate the spontaneous evolution of more complex computational pathways.

**Experimenting with Stochastic GoL:** Another future direction involves experimenting with stochastic versions of the Game of Life. In a stochastic GoL, state transitions incorporate elements of randomness, which can introduce variability and potentially enhance the robustness and adaptability of the model. By integrating stochasticity, Gol-LM can explore a broader range of behaviors and solutions, further enriching its learning and generalization capabilities. Furthermore, it allows to remove the random mutation part of the genetic algorithm and let the model internally organize in order to mutate in the most efficient way and keep itself on the criticality boundary where randomness (mutations) and order are balanced in the optimal way to increase reward faster.

## Installation and Usage

The installation process is manual and you need to clone/download this repository and execute it in your python environment.
During execution a popup window will appear where you can visualize the inference process.
The training part is not implemented yet.

### Requirements
The requirements are the standard ones for scientific computing: Numpy, Matplotlib.
You also need the TkAgg interface for matplotlib (in some cases it may be not installed automatically).

### Files
The files provided are python files or jupyter notebooks. Python files contain the core logic and classes of Gol-LM to be used as libraries and imported in jupyter notebook files where we perform tests.

- gol_lm_base.py       -> Base implementation, without meta-learning and reward settings
- gol_lm_metalearn.py  -> Experimental implementation, with meta-learning
- test_base.ipynb      -> Test for the base implementation
- test_metalearn.ipynb -> Test for the meta-learning setting 
