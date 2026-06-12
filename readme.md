### Agent Adaptive V5

This agent aims to implement an agent that is robust in the case where the opponent performs very suboptimal and to cases where the opponent plays at least as good as itself. Tries to get rid of the naiv assumption. Experiments have shown that this Adaptive Approach performs not better then the naive agents.

The agent tries to calculate a goodness score which indicates how good the other opponent is from zero to one. If the agents goodness score is near 1 it should behave like the ensemble-naiv-heuristic and if score is near 0 should incorporate more randomness to the planning and simulation rollouts. 

The goodness score is approximation of the likelihood that the other agent is the adaptive agent itself. It uses the heuristic as a quick approximation it's own policy. Then this likelihood is smoothed over the last few iteration by a moving average which is optimistically initiated.

This score is used during the selection phase and the rollout simulation. During selection for all tree nodes which correspond to the opponent  a random node is selected with probability (1-goodness). And during the rollout phase at each step where the opponent performs an action a random one is chosen with a probability of  1-goodness otherwise the heuristic is used to guide the rollout.

Originally this approach was indeed performing better than the naiv approaches but by introducing more randomness into the naive approaches the gap was closed and the naiv heuristic ensemble agent performed in the experiments the best 