/*
 * Purpose: Internal determinized sub-agent used by AgentAdaptiveV5 ensemble searches.
 */
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

    float getGoodness() {
        return goodness;
    }

    void setGoodness(float goodness) {
        this.goodness = goodness;
    }

    float getRolloutGreedyProbability() {
        return rolloutGreedyProbability;
    }

    HeuristicConfiguration getHeuristic() {
        return heuristic;
    }

    int getPlayerId() {
        return playerId;
    }

    void setPlayerId(int playerId) {
        this.playerId = playerId;
    }

    void resetPartnerModel() {
        goodness = 1.0f;
        lastEvaluatedIndex = 0;
        historyState = null;
        partnerActionCount = 0;
        partnerLikelihoodSum = 0.0f;
    }

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

        float averageLikelihood = partnerLikelihoodSum / partnerActionCount;
        float optimisticAverage = (((partnerLikelihoodSum + GOODNESS_PRIOR_WEIGHT * GOODNESS_PRIOR_MEAN)  / (partnerActionCount + GOODNESS_PRIOR_WEIGHT))+averageLikelihood)/2.0f;
        if(optimisticAverage > 0.4){
            goodness = Math.min(1.0f,0.4f+ optimisticAverage * 1.6f);
        }
        else{
            goodness = Math.max(0, optimisticAverage);
        }
    }

    private void evaluatePartnerAction(CarcassonneAction action, long computationTime, TimeUnit timeUnit) {
        ActionSet actions = historyState.calculatePossibleActionsUnique();
        if (actions.isEmpty()) {
            return;
        }

        float likelihood = evaluateHeuristicPolicyLikelihood(historyState, actions, action.getValue());

        partnerActionCount++;
        partnerLikelihoodSum += likelihood;
    }

    private float evaluateHeuristicPolicyLikelihood(State state, ActionSet actions, int actionValue) {
        if (actions.size() == 1) {
            return 1.0f;
        }

        float[] priors = computePriors(state, actions);
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

    private float[] computePriors(State state, ActionSet actions) {
        float[] priors = new float[actions.size()];
        int cachedPositionRotation = Integer.MIN_VALUE;
        float cachedTileScore = 0f;

        for (int i = 0; i < actions.size(); i++) {
            int candidateAction = actions.get(i);
            int x = CarcassonneActionLayoutBit.getX(candidateAction);
            int y = CarcassonneActionLayoutBit.getY(candidateAction);
            int rotation = CarcassonneActionLayoutBit.getRotation(candidateAction);
            int positionRotationKey = (x << 10) ^ (y << 2) ^ rotation;

            if (positionRotationKey != cachedPositionRotation) {
                cachedPositionRotation = positionRotationKey;
                cachedTileScore = HeuristicManager.tilePlacementScore(state, x, y, rotation, heuristic.positionHeuristik());
            }

            priors[i] = HeuristicManager.computePrior(state, candidateAction, cachedTileScore, heuristic);
        }
        return priors;
    }

    private int findMatchingIndex(ActionSet actions, int actionValue) {
        for (int i = 0; i < actions.size(); i++) {
            if (actions.get(i) == actionValue) {
                return i;
            }
        }
        return -1;
    }


    int rolloutCount() {
        return config.rollouts();
    }

    @Override
    protected DeterminizedActionNodePUCTV5 rootFactory(State initialState) {
        return new DeterminizedActionNodePUCTV5(this, null, 0, initialState, 0f, config.c(), heuristic);
    }

    @Override
    public DeterminizedActionNodePUCTV5 childFactory(DeterminizedActionNodePUCTV5 parent, int action, State checkpoint) {
        return new DeterminizedActionNodePUCTV5(this, parent, action, checkpoint, 0f, config.c(), heuristic);
    }
}
