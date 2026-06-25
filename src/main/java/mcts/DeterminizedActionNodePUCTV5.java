package mcts;

import abstractDeterminized.AbstractDeterminizedAgent;
import abstractDeterminized.AbstractPuctActionNode;
import model.collections.ActionSet;
import model.heuristic.HeuristicConfiguration;
import model.state.State;

/// DeterminizedActionNodePUCTV5
///
/// PUCT-based action node for the AdaptiveV5 agent. Modifies selection and rollout action
/// choices according to the partner agent goodness modeling. Depending on the goodness value
/// randomness is introduced to the selction process and the simulation
public class DeterminizedActionNodePUCTV5 extends AbstractPuctActionNode<DeterminizedActionNodePUCTV5> {
    private int turnPlayer = -1;

    /// DeterminizedActionNodePUCTV5
    ///
    /// @param agent the determinized agent managing the search tree
    /// @param parent the parent node in the search tree
    /// @param action the action represented by this node
    /// @param checkpoint the checkpointed board state (only stored at
    /// checkpoints/root)
    /// @param heuristic the prior heuristic probability/score of this node action
    /// @param explorationCoefficient the constant controlling exploration vs
    /// exploitation
    /// @param heuristicConfiguration the heuristic configuration used for rollout
    /// evaluation
    public DeterminizedActionNodePUCTV5(
            AbstractDeterminizedAgent<DeterminizedActionNodePUCTV5> agent,
            DeterminizedActionNodePUCTV5 parent,
            int action,
            State checkpoint,
            float heuristic,
            float explorationCoefficient,
            HeuristicConfiguration heuristicConfiguration) {
        super(agent, parent, action, checkpoint, heuristic, explorationCoefficient, heuristicConfiguration);
        if (parent == null && checkpoint != null) {
            this.turnPlayer = checkpoint.getCurrentPlayer();
        }
    }

    /// select
    ///
    /// Performs node selection. Incorporates partner modeling where opponent moves
    /// may be
    /// simulated as random if partner goodness score is low.
    ///
    /// @return the selected child node
    @Override
    public DeterminizedActionNodePUCTV5 select() {
        if (children == null || children.length == 0) {
            return self();
        }

        AdaptiveV5DeterminizedAgent agentV5 = (AdaptiveV5DeterminizedAgent) agent;
        DeterminizedActionNodePUCTV5 best;
        if (turnPlayer != agentV5.getPlayerId()) {
            if (agent.rand.nextFloat() >= agentV5.getGoodness()) {
                best = children[agent.rand.nextInt(children.length)];
            } else {
                best = selectUnvisitedOrBestUsb(children, children.length);
            }
        } else {
            best = selectUnvisitedOrBestUsb(children, children.length);
        }
        assert best != null;
        return best.getVisits() == 0 ? best : best.select();
    }

    /// expand
    ///
    /// Expands the current node initializing children with heuristic priors.
    ///
    /// @param state the current board state
    /// @return the expanded node
    @Override
    public DeterminizedActionNodePUCTV5 expand(State state) {
        if (children != null || state.isGameOver()) {
            return self();
        }

        this.turnPlayer = state.getCurrentPlayer();
        return expandWithHeuristicScores(
                state,
                false,
                (childAction, prior) -> new DeterminizedActionNodePUCTV5(agent, this, childAction, null, prior,
                        explorationCoefficient, heuristicConfiguration));
    }

    /// simulate
    ///
    /// Performs simulations/rollouts from the current node using a combination of
    /// random
    /// and greedy heuristic action selections adjusted by goodness scores.
    ///
    /// @param state the board state to start the simulation from
    /// @return the evaluated reward/score from the simulation
    @Override
    public float simulate(State state) {
        AdaptiveV5DeterminizedAgent agentV5 = (AdaptiveV5DeterminizedAgent) agent;
        return averageHeuristicRollout(state, agentV5.rolloutCount(), (rolloutState, actions) -> {
            float prob = agentV5.getRolloutGreedyProbability();
            if (rolloutState.getCurrentPlayer() != agentV5.getPlayerId()) {
                prob = agentV5.getGoodness() * prob;
            }

            if (agent.rand.nextFloat() < prob) {
                return chooseGreedyHeuristicAction(rolloutState, actions, 0.017);
            } else {
                return actions.get(agent.rand.nextInt(actions.size()));
            }
        });
    }

    /// newChildrenArray
    ///
    /// Helper factory method to instantiate the children node array.
    ///
    /// @param size the size of the array
    /// @return the created array of children nodes
    @Override
    protected DeterminizedActionNodePUCTV5[] newChildrenArray(int size) {
        return new DeterminizedActionNodePUCTV5[size];
    }

}
