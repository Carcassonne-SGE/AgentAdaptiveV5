### Agent Adaptive V5

The setting of the Collaborational Carcassonne game is that two agent try togehter to maximize their combined score. The only allowed assumption is that the other agent does not actively try reduce the score. Under this assumption the worst a agent can perform is to play completely random. The Naive Agent Approaches do not take into account that the other agent may be worse. Theoretically this means that it could happen that the Naive Agents plan for a strategy where the other player has to participate in a smart manner but the agent is not competent enough to do so. This could theoretically reduce the overall score when playing with a random agent. 

The AdaptiveV5 Agent tries to be more robust against an weaker opponent. The agent tries to model the performance of the other agent with a goodness score which is then used to internally model the beheaviour of the other agent during MCTS

The goodness score is a approximation of the likelihood that the other agent is the adaptive agent itself. It uses the heuristic as a quick approximation of it's own policy. This likelihood approximation is very noisy so it is smoothed over the last few steps by a moving average which is optimistically initiated.

This score is used during the MCTS selection phase and the simulation phase. When selecting a node in the search tree relatiting to an action of the opponent, with a chance of (1-goodness) a random node is chosen otherwise using the normal PUCT procedure. During simulation the opponent actions are selected using the heuristic with a chance of (goodness) otherwise a random actions is chosen.
.
Originally this approach was indeed performing better than the naive approaches but by introducing more randomness into the naive approaches the gap was closed and the naive heuristic ensemble agent performed in the experiments the best even against the random agent 