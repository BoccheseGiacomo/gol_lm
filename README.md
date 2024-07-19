# Gol-LM
A Language Model based on Game Of Life Cellular Automata with potential meta learning capabilities.

*Gol-LM © 2024 by Giacomo Bocchese is licensed under CC BY-NC-SA 4.0 with additional conditions specified in the LICENSE file.*

## Abstract
Gol-LM is an experimental research project focused on developing an adaptive language model using the principles of the Game of Life (GoL) cellular automaton. By harnessing the Turing completeness of GoL (finite memory approximation), Gol-LM enables the formation and evolution of sophisticated internal algorithms within a two-dimensional grid. The model employs genetic algorithms for evolutionary optimization, positioning Gol-LM as a robust and flexible language model capable of complex computational tasks.

Distinctive to Gol-LM is its ability to achieve spontaneous meta-learning, facilitating the emergence of self-reinforcement learning and dynamic memory organization. This capability allows the model to autonomously evolve internal optimization strategies, adapting its behavior based on external rewards. This mirrors natural processes where complexity and adaptability arise from non-linear, interactive systems that evolve towards optimal solutions.

As a language model, Gol-LM translates input sequences into coherent language predictions through iterative evolution and adaptation. The model's learning process is guided by external feedback, which informs its internal state adjustments and enhances its predictive accuracy over time.

It is important to note that Gol-LM is in a highly experimental stage. Many theoretical aspects are derived from rigorous deduction and intuition based on known principles in deep learning and emergent systems. The forthcoming phases of the project are exploratory, aiming to systematically uncover the model's capabilities and constraints. These steps are essential for transitioning from theoretical constructs to empirical results, advancing our understanding of cellular automata-based language modeling.

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

![gollmpreview](https://github.com/user-attachments/assets/339917d0-55de-4d41-9a30-a993c80bbdf3)


<div>
  <img src="https://github.com/user-attachments/assets/f1773f5a-1ddb-4e8c-9bde-c93dec488601" alt="Gol-LM Simulation" width="auto" height="auto">
  <p><em>Figure 1: A showing the inference process of a random, non trained, Gol-LM.</em></p>
  <p><em>State space dimension: 61x50 = 3050 cells</em></p>
  <p><em>Vocabulary dimension: 11 ; Symbols : ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '|']</em></p>
  <br>
</div>
