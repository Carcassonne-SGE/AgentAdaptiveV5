package mcts;

import abstractDeterminized.AbstractDeterminizedAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import model.bits.CarcassonneActionLayoutBit;
import model.collections.ActionSet;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;
import model.state.State;
import sge.CarcassonneAction;
import sge.CarcassonneGame;

import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/// AdaptiveV5DeterminizedAgent
///
/// Internal determinized sub-agent used by AgentAdaptiveV5 ensemble searches.
/// performs the goognes modeling and modified search
/// 
class AdaptiveV5DeterminizedAgent extends AbstractDeterminizedAgent<DeterminizedActionNodePUCTV5> {
    private static final float GOODNESS_PRIOR_MEAN = 1.0f;
    private static final int GOODNESS_PRIOR_WEIGHT = 2;

    private final HeuristicConfiguration heuristic;
    private final float rolloutGreedyProbability;
    private float goodness = 1.0f;
    private int lastEvaluatedIndex = 0;
    private State historyState;
    private int partnerActionCount = 0;
    private float partnerLikelihoodSum = 0.0f;

    /// AdaptiveV5DeterminizedAgent
    ///
    /// Constructor for AdaptiveV5DeterminizedAgent.
    ///
    /// @param logger the logger instance used for reporting
    /// @param rand the random number generator
    /// @param config the configuration settings for the agent
    /// @param heuristic the heuristic configuration used for search and evaluation
    /// @param rolloutGreedyProbability the probability of choosing a greedy
    /// heuristic action during rollouts
    AdaptiveV5DeterminizedAgent(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic,
            float rolloutGreedyProbability) {
        super(logger, config, rand);
        this.heuristic = heuristic == null ? HeuristicManager.createDefaultHeuristic() : heuristic;
        this.rolloutGreedyProbability = rolloutGreedyProbability;
    }

    /// resetPartnerModel
    ///
    /// Resets the internal statistics used to model and evaluate the partner agent.
    void resetPartnerModel() {
        goodness = 1.0f;
        lastEvaluatedIndex = 0;
        historyState = null;
        partnerActionCount = 0;
        partnerLikelihoodSum = 0.0f;
    }

    /// updateGoodness
    ///
    /// Processes action records to evaluate the partner agent actions and update
    /// its goodness score.
    ///
    /// @param game the Carcassonne game instance
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    void updateGoodness(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        List<at.ac.tuwien.ifs.sge.game.ActionRecord<CarcassonneAction>> records = game.getActionRecords();

        if (historyState == null) {
            historyState = new State(game.getBoard().getGameConfig());
            lastEvaluatedIndex = 0;
            partnerActionCount = 0;
            partnerLikelihoodSum = 0.0f;
        }

        for (int idx = lastEvaluatedIndex; idx < records.size(); idx++) {
            at.ac.tuwien.ifs.sge.game.ActionRecord<CarcassonneAction> record = records.get(idx);
            CarcassonneAction action = record.getAction();
            if (action.isAction() && record.getPlayer() != playerId) {
                evaluatePartnerAction(action, computationTime, timeUnit);
            }
            historyState.doAction(action);
        }

        lastEvaluatedIndex = records.size();
        if (partnerActionCount == 0) {
            goodness = 1.0f;
            return;
        }

        float rawAverage = partnerLikelihoodSum / partnerActionCount;
        float priorAverage = (partnerLikelihoodSum + GOODNESS_PRIOR_WEIGHT * GOODNESS_PRIOR_MEAN)
                / (partnerActionCount + GOODNESS_PRIOR_WEIGHT);
        float optimisticAverage = (priorAverage + rawAverage) / 2.0f;
        if (optimisticAverage > 0.4f) {
            goodness = Math.min(1.0f, 0.4f + optimisticAverage * 1.6f);
        } else {
            goodness = Math.max(0f, optimisticAverage);
        }
    }

    /// evaluatePartnerAction
    ///
    /// Evaluates a single action taken by the partner agent computing its
    /// likelihood
    /// under our heuristic policy model.
    ///
    /// @param action the action taken by the partner
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    private void evaluatePartnerAction(CarcassonneAction action, long computationTime, TimeUnit timeUnit) {
        ActionSet actions = historyState.calculatePossibleActionsUnique();
        if (actions.isEmpty()) {
            return;
        }
        float likelihood = evaluateHeuristicPolicyLikelihood(historyState, actions, action.getValue());
        partnerActionCount++;
        partnerLikelihoodSum += likelihood;
    }

    /// evaluateHeuristicPolicyLikelihood
    ///
    /// Computes the likelihood of a chosen action based on the prior scores
    /// computed by the heuristic policy.
    ///
    /// @param state the state prior to taking the action
    /// @param actions the set of legal actions
    /// @param actionValue the integer value of the chosen action
    /// @return the likelihood score
    private float evaluateHeuristicPolicyLikelihood(State state, ActionSet actions, int actionValue) {
        if (actions.size() == 1) {
            return 1.0f;
        }

        float[] priors = HeuristicManager.computePriors(state, actions, heuristic);
        int matchingIndex = findMatchingIndex(actions, actionValue);
        if (matchingIndex < 0) {
            return 0f;
        }
        float min = Float.POSITIVE_INFINITY;
        for (float prior : priors) {
            min = Math.min(min, prior);
        }

        double totalWeight = 0.0;
        double chosenOrWorseWeight = 0.0;
        float chosenPrior = priors[matchingIndex];

        for (float prior : priors) {
            double weight = prior - min;
            totalWeight += weight;
            if (prior <= chosenPrior) {
                chosenOrWorseWeight += weight;
            }
        }
        if (totalWeight <= 0.0) {
            return 1.0f;
        }

        return (float) (chosenOrWorseWeight / totalWeight);
    }

    /// findMatchingIndex
    ///
    /// Finds the index of a specific action value within the ActionSet.
    ///
    /// @param actions the set of legal actions
    /// @param actionValue the action value to locate
    /// @return the index or -1 if not found
    private int findMatchingIndex(ActionSet actions, int actionValue) {
        for (int i = 0; i < actions.size(); i++) {
            if (actions.get(i) == actionValue) {
                return i;
            }
        }
        return -1;
    }

    /// rootFactory
    ///
    /// Factory method to construct the root node of the MCTS tree.
    ///
    /// @param initialState the initial game state
    /// @return the constructed root node
    @Override
    protected DeterminizedActionNodePUCTV5 rootFactory(State initialState) {
        return new DeterminizedActionNodePUCTV5(this, null, 0, initialState, 0f, config.c(), heuristic);
    }

    /// childFactory
    ///
    /// Factory method to construct a new child node in the MCTS tree.
    ///
    /// @param parent the parent node
    /// @param action the action taken
    /// @param checkpoint the checkpointed state of the child node
    /// @return the constructed child node
    @Override
    public DeterminizedActionNodePUCTV5 childFactory(DeterminizedActionNodePUCTV5 parent, int action,
            State checkpoint) {
        return new DeterminizedActionNodePUCTV5(this, parent, action, checkpoint, 0f, config.c(), heuristic);
    }

    /// getGoodness
    ///
    /// Returns the partner agents evaluated goodness value.
    float getGoodness() {
        return goodness;
    }

    /// setGoodness
    ///
    /// Sets the partner agent evaluated goodness value.
    void setGoodness(float goodness) {
        this.goodness = goodness;
    }

    /// getRolloutGreedyProbability
    ///
    /// Returns the rollout greedy probability.
    float getRolloutGreedyProbability() {
        return rolloutGreedyProbability;
    }

    /// getPlayerId
    ///
    /// Returns the player ID of this agent.
    int getPlayerId() {
        return playerId;
    }

    /// setPlayerId
    ///
    /// Sets the player ID of this agent.
    void setPlayerId(int playerId) {
        this.playerId = playerId;
    }

    /// rolloutCount
    ///
    /// Returns the number of rollouts to perform per simulation.
    ///
    /// @return the rollout count
    int rolloutCount() {
        return config.rollouts();
    }

}
