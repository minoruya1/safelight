# SafeLight
This is the implementation trace of SafeLight, which is accepted by AAAI'23. Here's the paper: [SafeLight: A Reinforcement Learning Method toward Collision-free Traffic Signal Control](https://arxiv.org/abs/2211.10871).

The safety module is built upon on existing RL-based traffic signal control method. The paper can be found here: [A Deep Reinforcement Learning Network for Traffic Light Cycle Control](https://web.njit.edu/~gwang/papers/2019ITVT.pdf), and the code can be found here: https://github.com/Ring367/A-Deep-Reinforcement-Learning-Network-for-Traffic-Light-Cycle-Control.

We use the benchmark environment [RESCO](https://github.com/Pi-Star-Lab/RESCO) to evaluate the performance of our proposed method.

Please see the documentation of [SUMO](https://sumo.dlr.de/docs/index.html) the environment simulator as the first step to model the intersection. These are useful links:

- TraCI: traffic control interface
https://sumo.dlr.de/docs/TraCI.html
- Create Collisions: https://sumo.dlr.de/docs/Simulation/Safety.html


## System Configuration
- tensorflow version 1.15.0
- numpy version 1.19.2
- scipy version 1.5.2
- pandas version 0.25.3
- sumolib version 1.15.0
- traci version 1.15.0
- matplotlib version 3.3.4
