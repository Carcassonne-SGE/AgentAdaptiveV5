package mcts;

import abstractDeterminized.AbstractDeterminizedAgent;
import abstractDeterminized.AbstractDeterminizedEnsembleAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;
import sge.CarcassonneAction;
import sge.CarcassonneGame;

import java.util.Random;
import java.util.concurrent.TimeUnit;

public class AgentAdaptiveV5 extends AbstractDeterminizedEnsembleAgent {
    public static final int ENSEMBLE_SIZE = 5;

    private final HeuristicConfiguration heuristic;
    private final float rolloutGreedyProbability;

    public AgentAdaptiveV5() {
        this((Logger) null);
    }

    public AgentAdaptiveV5(Logger logger) {
        this(
                logger,
                new Random(99),
                new AbstractAgentConfiguration(1.4f, 20, 0.2f, 5),
                HeuristicManager.createDefaultHeuristic(),
                0.4f
        );
    }

    public AgentAdaptiveV5(Random rand) {
        this(
                null,
                rand,
                new AbstractAgentConfiguration(1.4f, 20, 0.2f, 5),
                HeuristicManager.createDefaultHeuristic(),
                0.4f
        );
    }

    public AgentAdaptiveV5(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic
    ) {
        this(logger, rand, config, heuristic, 0.4f);
    }

    public AgentAdaptiveV5(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic,
            float rolloutGreedyProbability
    ) {
        super(
                logger,
                config,
                ENSEMBLE_SIZE,
                (subLogger, subConfig, subRand) -> new AdaptiveV5DeterminizedAgent(
                        subLogger,
                        subRand,
                        subConfig,
                        heuristic,
                        rolloutGreedyProbability
                ),
                rand
        );
        this.heuristic = heuristic == null ? HeuristicManager.createDefaultHeuristic() : heuristic;
        this.rolloutGreedyProbability = rolloutGreedyProbability;
    }

    @Override
    public void setUp(int numberOfPlayers, int playerNumber) {
        super.setUp(numberOfPlayers, playerNumber);
        for (abstractDeterminized.AbstractDeterminizedAgent<?> agent : agents) {
            AdaptiveV5DeterminizedAgent adaptiveAgent = (AdaptiveV5DeterminizedAgent) agent;
            adaptiveAgent.setPlayerId(playerNumber);
            adaptiveAgent.resetPartnerModel();
        }
    }

    @Override
    public CarcassonneAction computeNextAction(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        AdaptiveV5DeterminizedAgent referenceAgent = (AdaptiveV5DeterminizedAgent) agents[0];
        referenceAgent.updateGoodness(game, computationTime, timeUnit);
        float goodness = referenceAgent.getGoodness();
        for (abstractDeterminized.AbstractDeterminizedAgent<?> agent : agents) {
            ((AdaptiveV5DeterminizedAgent) agent).setGoodness(goodness);
        }
        System.out.println("AdaptiveV5 Player " + playerId + " partner goodness score: " + goodness);
        if (logger != null) {
            logger._infof("AdaptiveV5 Player %d partner goodness score: %.4f", playerId, goodness);
        }
        return super.computeNextAction(game, computationTime, timeUnit);
    }

    HeuristicConfiguration getHeuristic() {
        return heuristic;
    }

    float getRolloutGreedyProbability() {
        return rolloutGreedyProbability;
    }
}
