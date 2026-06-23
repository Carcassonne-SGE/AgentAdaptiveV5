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

/// AgentAdaptiveV5
///
/// An adaptive ensemble agent that adjusts its search strategy based on the evaluated
/// skill/goodness of the opponent.
public class AgentAdaptiveV5 extends AbstractDeterminizedEnsembleAgent {
    public static final int ENSEMBLE_SIZE = 5;

    private final HeuristicConfiguration heuristic;
    private final float rolloutGreedyProbability;

    /// AgentAdaptiveV5
    ///
    /// Default constructor creating an agent with no logger.
    public AgentAdaptiveV5() {
        this((Logger) null);
    }

    /// AgentAdaptiveV5
    ///
    /// Constructor creating an agent with a logger and default configuration.
    ///
    /// @param logger the logger instance
    public AgentAdaptiveV5(Logger logger) {
        this(
                logger,
                new Random(99),
                new AbstractAgentConfiguration(1.4f, 20, 0.2f, 5),
                HeuristicManager.createDefaultHeuristic(),
                0.4f
        );
    }

    /// AgentAdaptiveV5
    ///
    /// Constructor creating an agent with a specific random generator and default configuration.
    ///
    /// @param rand the random number generator
    public AgentAdaptiveV5(Random rand) {
        this(
                null,
                rand,
                new AbstractAgentConfiguration(1.4f, 20, 0.2f, 5),
                HeuristicManager.createDefaultHeuristic(),
                0.4f
        );
    }

    /// AgentAdaptiveV5
    ///
    /// Constructor creating an agent with specified configurations and defaults for rollout probability.
    ///
    /// @param logger the logger instance
    /// @param rand the random number generator
    /// @param config the agent configuration
    /// @param heuristic the heuristic configuration
    public AgentAdaptiveV5(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic
    ) {
        this(logger, rand, config, heuristic, 0.4f);
    }

    /// AgentAdaptiveV5
    ///
    /// Constructor creating an agent with full custom configurations.
    ///
    /// @param logger the logger instance
    /// @param rand the random number generator
    /// @param config the agent configuration
    /// @param heuristic the heuristic configuration
    /// @param rolloutGreedyProbability the greedy probability for rollouts
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

    /// setUp
    ///
    /// Sets up the ensemble agent, player ID, and resets partner models for sub-agents.
    ///
    /// @param numberOfPlayers the total number of players
    /// @param playerNumber the player ID assigned to this agent
    @Override
    public void setUp(int numberOfPlayers, int playerNumber) {
        super.setUp(numberOfPlayers, playerNumber);
        for (abstractDeterminized.AbstractDeterminizedAgent<?> agent : agents) {
            AdaptiveV5DeterminizedAgent adaptiveAgent = (AdaptiveV5DeterminizedAgent) agent;
            adaptiveAgent.setPlayerId(playerNumber);
            adaptiveAgent.resetPartnerModel();
        }
    }

    /// computeNextAction
    ///
    /// Evaluates and updates the opponent/partner's goodness model, shares it among
    /// ensemble sub-agents, and computes the next best action.
    ///
    /// @param game the Carcassonne game instance
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    /// @return the selected CarcassonneAction
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

    /// getHeuristic
    ///
    /// Returns the heuristic configuration.
    ///
    /// @return the heuristic configuration
    HeuristicConfiguration getHeuristic() {
        return heuristic;
    }

    /// getRolloutGreedyProbability
    ///
    /// Returns the rollout greedy probability.
    ///
    /// @return the rollout greedy probability
    float getRolloutGreedyProbability() {
        return rolloutGreedyProbability;
    }
}
