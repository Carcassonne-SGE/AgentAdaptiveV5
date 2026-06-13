package mcts;

import abstractDeterminized.AbstractDeterminizedAgent;
import abstractDeterminized.AbstractPuctActionNode;
import model.collections.ActionSet;
import model.heuristic.HeuristicConfiguration;
import model.state.State;


public class DeterminizedActionNodePUCTV5 extends AbstractPuctActionNode<DeterminizedActionNodePUCTV5> {
    private int turnPlayer = -1;

   
    public DeterminizedActionNodePUCTV5(
            AbstractDeterminizedAgent<DeterminizedActionNodePUCTV5> agent,
            DeterminizedActionNodePUCTV5 parent,
            int action,
            State checkpoint,
            float heuristic,
            float explorationCoefficient,
            HeuristicConfiguration heuristicConfiguration
    ) {
        super(agent, parent, action, checkpoint, heuristic, explorationCoefficient, heuristicConfiguration);
        if (parent == null && checkpoint != null) {
            this.turnPlayer = checkpoint.getCurrentPlayer();
        }
    }

    @Override
    public float getSelectionScore() {
        if (visits == 0) {
            return Float.MAX_VALUE;
        }
        float q = agent.ucbTransform(value / (float) visits);
        if (parent == null || parent.getVisits() <= 1) {
            return q;
        }
        float b = (float) (Math.sqrt(parent.getVisits()) / (1 + visits));
        return q + explorationCoefficient * heuristic * b;
    }

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

    @Override
    protected DeterminizedActionNodePUCTV5[] newChildrenArray(int size) {
        return new DeterminizedActionNodePUCTV5[size];
    }

    @Override
    public DeterminizedActionNodePUCTV5 expand(State state) {
        if (children != null || state.isGameOver()) {
            return self();
        }

        this.turnPlayer = state.getCurrentPlayer();
        return expandWithHeuristicScores(
                state,
                false,
                (childAction, prior) -> new DeterminizedActionNodePUCTV5(agent, this, childAction, null, prior, explorationCoefficient, heuristicConfiguration)
        );
    }

    @Override
    public float simulate(State state) {
        AdaptiveV5DeterminizedAgent agentV5 = (AdaptiveV5DeterminizedAgent) agent;
        return averageHeuristicRollout(state, agentV5.rolloutCount(), (rolloutState, actions) -> chooseRolloutAction(rolloutState, actions, agentV5));
    }

    private int chooseRolloutAction(State state, ActionSet actions, AdaptiveV5DeterminizedAgent agentV5) {
        float prob = agentV5.getRolloutGreedyProbability();
        if (state.getCurrentPlayer() != agentV5.getPlayerId()) {
            prob = agentV5.getGoodness() * prob;
        }

        if (agent.rand.nextFloat() < prob) {
            return chooseGreedyHeuristicAction(state, actions, 0.017);
        } else {
            return actions.get(agent.rand.nextInt(actions.size()));
        }
    }
}
