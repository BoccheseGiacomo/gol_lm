# Gol-LM
A Language Model based on Game Of Life Cellular Automata with potential meta learning capabilities.

*Gol-LM © 2024 by Giacomo Bocchese is licensed under CC BY-NC-SA 4.0 with additional conditions specified in the LICENSE file.*

## Abstract
Gol-LM is an experimental research project focused on developing an adaptive language model using the principles of the Game of Life (GoL) cellular automaton. By harnessing the Turing completeness of GoL (finite memory approximation), Gol-LM enables the formation and evolution of sophisticated internal algorithms within a two-dimensional grid. The model employs genetic algorithms for evolutionary optimization, positioning Gol-LM as a robust and flexible language model capable of complex computational tasks.

Distinctive to Gol-LM is its ability to achieve spontaneous meta-learning, facilitating the emergence of self-reinforcement learning and dynamic memory organization. This capability allows the model to autonomously evolve internal optimization strategies, adapting its behavior based on external rewards. This mirrors natural processes where complexity and adaptability arise from non-linear, interactive systems that evolve towards optimal solutions.

As a language model, Gol-LM translates input sequences into coherent language predictions through iterative evolution and adaptation. The model's learning process is guided by external feedback, which informs its internal state adjustments and enhances its predictive accuracy over time.

It is important to note that Gol-LM is in a highly experimental stage. Many theoretical aspects are derived from rigorous deduction and intuition based on known principles in deep learning and emergent systems. The forthcoming phases of the project are exploratory, aiming to systematically uncover the model's capabilities and constraints. These steps are essential for transitioning from theoretical constructs to empirical results, advancing our understanding of cellular automata-based language modeling.

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

**Evolutionary Training**: The training process of Gol-LM relies on genetic algorithms, emphasizing the exploration of diverse initial states and evolutionary strategies. This approach aligns with the non-differentiable nature of GoL and facilitates the iterative improvement of internal algorithms. Through the selection of the best-performing states and the application of genetic operations, Gol-LM evolves to enhance its language modeling capabilities over successive generations.

Gol-LM represents a cutting-edge exploration into the potential of cellular automata for advanced language modeling, combining principles of Turing completeness, meta-learning, and evolutionary optimization to push the boundaries of adaptive computational systems.


![gollmpreview](https://github.com/user-attachments/assets/339917d0-55de-4d41-9a30-a993c80bbdf3)
