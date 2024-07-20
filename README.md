# Gol-LM
A Language Model based on Game Of Life Cellular Automata with potential meta learning capabilities.

*Gol-LM © 2024 by Giacomo Bocchese is licensed under CC BY-NC-SA 4.0 with additional conditions specified in the LICENSE file.*

## Abstract
Gol-LM is an experimental research project focused on developing an adaptive language model using the principles of the Game of Life (GoL) cellular automaton. By harnessing the Turing completeness of GoL (finite memory approximation), Gol-LM enables the formation and evolution of sophisticated internal algorithms within a two-dimensional grid. The model employs genetic algorithms for evolutionary optimization, positioning Gol-LM as a robust and flexible language model capable of complex computational tasks.

Distinctive to Gol-LM is its ability to achieve spontaneous meta-learning, facilitating the emergence of self-reinforcement learning and dynamic memory organization. This capability allows the model to autonomously evolve internal optimization strategies, adapting its behavior based on external rewards. This mirrors natural processes where complexity and adaptability arise from non-linear, interactive systems that evolve towards optimal solutions.

As a language model, Gol-LM translates input sequences into coherent language predictions through iterative evolution and adaptation. The model's learning process is guided by external feedback, which informs its internal state adjustments and enhances its predictive accuracy over time.

It is important to note that Gol-LM is in a highly experimental stage. Many theoretical aspects are derived from rigorous deduction and intuition based on known principles in machine learning and emergent systems, but may not be proved enough. The forthcoming phases of the project are exploratory, aiming to systematically uncover the model's capabilities and constraints. These steps are essential for transitioning from theoretical constructs to empirical results, advancing our understanding of cellular automata-based language modeling.

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

**Evolutionary Training**: The training process of Gol-LM relies on genetic algorithms (or other black box optimization algorithms), emphasizing the exploration of diverse initial states and evolutionary strategies. This approach aligns with the non-differentiable nature of GoL and facilitates the iterative improvement of internal algorithms. Through the selection of the best-performing states and the application of genetic operations, Gol-LM evolves to enhance its language modeling capabilities over successive generations, and may learn to exploit reward information in order to build internal emergent optimizers.

Gol-LM represents a cutting-edge exploration into the potential of cellular automata for advanced language modeling, combining principles of Turing completeness, meta-learning, and evolutionary optimization to push the boundaries of adaptive computational systems.

**Not only language modeling**: The model is initially developed for language modeling tasks, but with minimal modifications it can be adapted to other settings, like driving an agent in RL settings.

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


