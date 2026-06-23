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
    /// @param rolloutGreedyProbability the probability of choosing a greedy heuristic action during rollouts
    AdaptiveV5DeterminizedAgent(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic,
            float rolloutGreedyProbability
    ) {
        super(logger, config, rand);
        this.heuristic = heuristic == null ? HeuristicManager.createDefaultHeuristic() : heuristic;
        this.rolloutGreedyProbability = rolloutGreedyProbability;
    }

    /// getGoodness
    ///
    /// Returns the partner agent's evaluated goodness value.
    ///
    /// @return the partner goodness value
    float getGoodness() {
        return goodness;
    }

    /// setGoodness
    ///
    /// Sets the partner agent's evaluated goodness value.
    ///
    /// @param goodness the new goodness value
    void setGoodness(float goodness) {
        this.goodness = goodness;
    }

    /// getRolloutGreedyProbability
    ///
    /// Returns the rollout greedy probability.
    ///
    /// @return the rollout greedy probability
    float getRolloutGreedyProbability() {
        return rolloutGreedyProbability;
    }

    /// getHeuristic
    ///
    /// Returns the heuristic configuration.
    ///
    /// @return the heuristic configuration
    HeuristicConfiguration getHeuristic() {
        return heuristic;
    }

    /// getPlayerId
    ///
    /// Returns the player ID of this agent.
    ///
    /// @return the player ID
    int getPlayerId() {
        return playerId;
    }

    /// setPlayerId
    ///
    /// Sets the player ID of this agent.
    ///
    /// @param playerId the player ID
    void setPlayerId(int playerId) {
        this.playerId = playerId;
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
    /// Processes action records to evaluate the partner agent's actions and update its goodness score.
    ///
    /// @param game the Carcassonne game instance
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    void updateGoodness(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        List<at.ac.tuwien.ifs.sge.game.ActionRecord<CarcassonneAction>> records = game.getActionRecords();

        // Initialize history tracking state starting from game start configuration
        if (historyState == null) {
            historyState = new State(game.getBoard().getGameConfig());
            lastEvaluatedIndex = 0;
            partnerActionCount = 0;
            partnerLikelihoodSum = 0.0f;
        }

        // Advance historyState action-by-action to catch up with newly made moves
        for (int idx = lastEvaluatedIndex; idx < records.size(); idx++) {
            at.ac.tuwien.ifs.sge.game.ActionRecord<CarcassonneAction> record = records.get(idx);
            CarcassonneAction action = record.getAction();
            // Evaluate only valid game-play actions made by the partner (not ourselves)
            if (action.isAction() && record.getPlayer() != playerId) {
                evaluatePartnerAction(action, computationTime, timeUnit);
            }
            historyState.doAction(action);
        }

        lastEvaluatedIndex = records.size();
        // If the partner has made no actions yet, assume maximum goodness default
        if (partnerActionCount == 0) {
            goodness = 1.0f;
            return;
        }

        // Calculate the raw average likelihood of partner's moves under our heuristic policy model
        float averageLikelihood = partnerLikelihoodSum / partnerActionCount;
        // Blend raw average with a prior mean to obtain an optimistic estimate (avoiding early pessimism)
        float optimisticAverage = (((partnerLikelihoodSum + GOODNESS_PRIOR_WEIGHT * GOODNESS_PRIOR_MEAN)  / (partnerActionCount + GOODNESS_PRIOR_WEIGHT))+averageLikelihood)/2.0f;
        // Apply thresholding and scaling: high compliance with policy yields goodness close to 1.0
        if(optimisticAverage > 0.4){
            goodness = Math.min(1.0f,0.4f+ optimisticAverage * 1.6f);
        }
        else{
            goodness = Math.max(0, optimisticAverage);
        }
    }

    /// evaluatePartnerAction
    ///
    /// Evaluates a single action taken by the partner agent, computing its likelihood
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

        // Evaluate the probability/likelihood of the partner's chosen action
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

        // Get prior scores for all legal actions
        float[] priors = computePriors(state, actions);
        int matchingIndex = findMatchingIndex(actions, actionValue);

        if (matchingIndex < 0) {
            return 0f;
        }

        // Shift priors so that the lowest score acts as 0 (relative differences matter)
        float min = Float.POSITIVE_INFINITY;
        for (float prior : priors) {
            min = Math.min(min, prior);
        }

        double totalWeight = 0.0;
        double chosenOrWorseWeight = 0.0;
        float chosenPrior = priors[matchingIndex];

        // Sum weights and compute the cumulative rank/probability weight of the partner's action
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

        // Returns proportion of action weight space that is as bad or worse than the selected action
        return (float) (chosenOrWorseWeight / totalWeight);
    }

    /// computePriors
    ///
    /// Computes the heuristic prior scores for all possible actions in the given state.
    ///
    /// @param state the current state
    /// @param actions the set of legal actions
    /// @return an array of float prior scores
    private float[] computePriors(State state, ActionSet actions) {
        float[] priors = new float[actions.size()];
        int cachedPositionRotation = Integer.MIN_VALUE;
        float cachedTileScore = 0f;

        for (int i = 0; i < actions.size(); i++) {
            int candidateAction = actions.get(i);
            // Unpack coordinate and rotation fields
            int x = CarcassonneActionLayoutBit.getX(candidateAction);
            int y = CarcassonneActionLayoutBit.getY(candidateAction);
            int rotation = CarcassonneActionLayoutBit.getRotation(candidateAction);
            int positionRotationKey = (x << 10) ^ (y << 2) ^ rotation;

            // Cache the tile placement score component to avoid redundant evaluation across areas/meeple choices
            if (positionRotationKey != cachedPositionRotation) {
                cachedPositionRotation = positionRotationKey;
                cachedTileScore = HeuristicManager.tilePlacementScore(state, x, y, rotation, heuristic.positionHeuristik());
            }

            priors[i] = HeuristicManager.computePrior(state, candidateAction, cachedTileScore, heuristic);
        }
        return priors;
    }

    /// findMatchingIndex
    ///
    /// Finds the index of a specific action value within the ActionSet.
    ///
    /// @param actions the set of legal actions
    /// @param actionValue the action value to locate
    /// @return the index, or -1 if not found
    private int findMatchingIndex(ActionSet actions, int actionValue) {
        for (int i = 0; i < actions.size(); i++) {
            if (actions.get(i) == actionValue) {
                return i;
            }
        }
        return -1;
    }

    /// rolloutCount
    ///
    /// Returns the number of rollouts to perform per simulation.
    ///
    /// @return the rollout count
    int rolloutCount() {
        return config.rollouts();
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
    public DeterminizedActionNodePUCTV5 childFactory(DeterminizedActionNodePUCTV5 parent, int action, State checkpoint) {
        return new DeterminizedActionNodePUCTV5(this, parent, action, checkpoint, 0f, config.c(), heuristic);
    }
}
