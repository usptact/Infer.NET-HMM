using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Utilities;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace FactorialHiddenMarkovModel
{
    /// <summary>
    /// Factorial Hidden Markov Model with multiple independent chains.
    /// 
    /// Model Structure:
    /// ================
    /// - Multiple HMM chains evolve independently in parallel
    /// - Each chain has its own states, transitions, and emission parameters
    /// - Observations are the SUM of emissions from ALL chains plus noise
    /// 
    /// Mathematical Formulation:
    /// =========================
    /// For C chains, each with states z^c_t:
    /// 
    ///   z^c_0 ~ Discrete(π^c)                    // Initial state for chain c
    ///   z^c_t | z^c_{t-1} ~ Discrete(A^c_{z^c_{t-1}})  // Transition for chain c
    ///   y_t | z^1_t,...,z^C_t ~ N(Σ_c μ^c_{z^c_t}, σ²_obs)  // Observation is sum
    /// 
    /// Where:
    /// - z^c_t: state of chain c at time t
    /// - π^c: initial state distribution for chain c
    /// - A^c: transition matrix for chain c
    /// - μ^c_k: emission mean for state k in chain c
    /// - σ²_obs: observation noise variance
    /// 
    /// Use Cases:
    /// ==========
    /// - Multi-factor time series (e.g., price = trend + seasonality + noise)
    /// - Signal decomposition (separate independent sources)
    /// - Multi-modal behavior (multiple independent processes affect observation)
    /// 
    /// Limitations:
    /// ============
    /// - Currently supports 1-3 chains (nested Switch blocks)
    /// - Inference is more challenging than standard HMM (exponential state space)
    /// - Requires careful prior specification for good results
    /// - Identifiability issues: multiple state combinations can produce same observation
    /// 
    /// Example Usage:
    /// ==============
    /// <code>
    /// // Create factorial HMM: 2 chains with 2 and 3 states
    /// int[] numStates = new int[] { 2, 3 };
    /// var model = new FactorialHiddenMarkovModel(sequenceLength: 100, numStates);
    /// 
    /// // Set priors and observe data
    /// model.SetPriors(initPriors, transPriors, meanPriors, precPriors, obsPrior);
    /// model.ObserveData(observations);
    /// model.InitialiseStatesRandomly();
    /// 
    /// // Run inference
    /// model.InferPosteriors();
    /// 
    /// // Get results
    /// int[][] states = model.GetMAPStates();
    /// double[][] means = model.GetEmissionMeans();
    /// double[][] contributions = model.DecomposeObservations();
    /// </code>
    /// </summary>
    public class FactorialHiddenMarkovModel
    {
        // Set up emission data
        private double[] EmitData;

        // Set up the ranges
        private int NumChains;
        private Range[] K;  // Number of states for each chain
        private Range T;    // Time steps

        // Set up model variables for each chain
        private Variable<int>[] ZeroStates;         // Initial states for each chain
        private VariableArray<int>[] ChainStates;   // States over time for each chain
        private VariableArray<double> Emissions;    // Observed emissions (sum of all chains)

        // Set up model parameters for each chain
        private Variable<Vector>[] ProbInit;        // Initial state distributions
        private VariableArray<Vector>[] CPTTrans;   // Transition probability tables
        private VariableArray<double>[] EmitMean;   // Emission means for each chain
        private VariableArray<double>[] EmitPrec;   // Emission precisions for each chain

        // Set up prior distributions for each chain
        private Variable<Dirichlet>[] ProbInitPrior;
        private VariableArray<Dirichlet>[] CPTTransPrior;
        private VariableArray<Gaussian>[] EmitMeanPrior;
        private VariableArray<Gamma>[] EmitPrecPrior;

        // Global observation precision (noise in the additive observation)
        private Variable<double> ObsPrec;
        private Variable<Gamma> ObsPrecPrior;

        // Set up model evidence
        private Variable<bool> ModelEvidence;

        // Inference engine
        private InferenceEngine Engine;

        // Set up posteriors
        public Dirichlet[] ProbInitPosterior;
        public Dirichlet[][] CPTTransPosterior;
        public Gaussian[][] EmitMeanPosterior;
        public Gamma[][] EmitPrecPosterior;
        public Discrete[][] ChainStatesPosterior;
        public Gamma ObsPrecPosterior;
        public Bernoulli ModelEvidencePosterior;

        /// <summary>
        /// Initializes a new instance of the Factorial HMM.
        /// </summary>
        /// <param name="chainLength">Length of the observation sequence.</param>
        /// <param name="numStatesPerChain">Array specifying number of states for each chain.</param>
        public FactorialHiddenMarkovModel(int chainLength, int[] numStatesPerChain)
        {
            if (numStatesPerChain == null || numStatesPerChain.Length < 2)
                throw new ArgumentException("Factorial HMM requires at least 2 chains. Use HiddenMarkovModel for single-chain models.", nameof(numStatesPerChain));
            
            if (numStatesPerChain.Length > 3)
                throw new ArgumentException("Currently supports maximum 3 chains", nameof(numStatesPerChain));
            
            NumChains = numStatesPerChain.Length;
            
            ModelEvidence = Variable.Bernoulli(0.5).Named("evidence");
            using (Variable.If(ModelEvidence))
            {
                T = new Range(chainLength).Named("T");
                
                // Initialize arrays for multiple chains
                K = new Range[NumChains];
                ZeroStates = new Variable<int>[NumChains];
                ChainStates = new VariableArray<int>[NumChains];
                ProbInit = new Variable<Vector>[NumChains];
                CPTTrans = new VariableArray<Vector>[NumChains];
                EmitMean = new VariableArray<double>[NumChains];
                EmitPrec = new VariableArray<double>[NumChains];
                ProbInitPrior = new Variable<Dirichlet>[NumChains];
                CPTTransPrior = new VariableArray<Dirichlet>[NumChains];
                EmitMeanPrior = new VariableArray<Gaussian>[NumChains];
                EmitPrecPrior = new VariableArray<Gamma>[NumChains];

                // Set up each chain independently
                for (int c = 0; c < NumChains; c++)
                {
                    K[c] = new Range(numStatesPerChain[c]).Named($"K{c}");
                    
                    // Initial state distribution
                    ProbInitPrior[c] = Variable.New<Dirichlet>().Named($"ProbInitPrior{c}");
                    ProbInit[c] = Variable<Vector>.Random(ProbInitPrior[c]).Named($"ProbInit{c}");
                    ProbInit[c].SetValueRange(K[c]);
                    
                    // Transition probability tables
                    CPTTransPrior[c] = Variable.Array<Dirichlet>(K[c]).Named($"CPTTransPrior{c}");
                    CPTTrans[c] = Variable.Array<Vector>(K[c]).Named($"CPTTrans{c}");
                    CPTTrans[c][K[c]] = Variable<Vector>.Random(CPTTransPrior[c][K[c]]);
                    CPTTrans[c].SetValueRange(K[c]);
                    
                    // Emission parameters
                    EmitMeanPrior[c] = Variable.Array<Gaussian>(K[c]).Named($"EmitMeanPrior{c}");
                    EmitMean[c] = Variable.Array<double>(K[c]).Named($"EmitMean{c}");
                    EmitMean[c][K[c]] = Variable<double>.Random(EmitMeanPrior[c][K[c]]);
                    
                    EmitPrecPrior[c] = Variable.Array<Gamma>(K[c]).Named($"EmitPrecPrior{c}");
                    EmitPrec[c] = Variable.Array<double>(K[c]).Named($"EmitPrec{c}");
                    EmitPrec[c][K[c]] = Variable<double>.Random(EmitPrecPrior[c][K[c]]);
                    
                    // Chain states
                    ZeroStates[c] = Variable.Discrete(ProbInit[c]).Named($"z0_{c}");
                    ChainStates[c] = Variable.Array<int>(T).Named($"Chain{c}States");
                }

                // Observation precision (noise in the sum)
                ObsPrecPrior = Variable.New<Gamma>().Named("ObsPrecPrior");
                ObsPrec = Variable<double>.Random(ObsPrecPrior).Named("ObsPrec");

                // Final observed emissions (sum of all chain contributions)
                Emissions = Variable.Array<double>(T).Named("Emissions");

                // Time evolution
                using (var block = Variable.ForEach(T))
                {
                    var t = block.Index;

                    // Evolve each chain's state
                    for (int c = 0; c < NumChains; c++)
                    {
                        var previousState = ChainStates[c][t - 1];

                        // Initial state (t=0)
                        using (Variable.If((t == 0).Named($"Initial{c}")))
                        {
                            using (Variable.Switch(ZeroStates[c]))
                            {
                                ChainStates[c][T] = Variable.Discrete(CPTTrans[c][ZeroStates[c]]);
                            }
                        }

                        // State transitions (t>0)
                        using (Variable.If((t > 0).Named($"Transition{c}")))
                        {
                            using (Variable.Switch(previousState))
                            {
                                ChainStates[c][t] = Variable.Discrete(CPTTrans[c][previousState]);
                            }
                        }
                    }

                    // For factorial HMM, we need to model the observation as sum of contributions
                    // We use a compound switch over all chain states
                    // For 2 chains with K0 and K1 states: we have K0*K1 combinations
                    
                    if (NumChains == 2)
                    {
                        // Two-chain factorial HMM
                        using (Variable.Switch(ChainStates[0][t]))
                        {
                            using (Variable.Switch(ChainStates[1][t]))
                            {
                                // Within nested switch blocks, we can safely add the means
                                Variable<double> sumOfMeans = 
                                    EmitMean[0][ChainStates[0][t]] + EmitMean[1][ChainStates[1][t]];
                                Emissions[t] = Variable.GaussianFromMeanAndPrecision(sumOfMeans, ObsPrec);
                            }
                        }
                    }
                    else if (NumChains == 3)
                    {
                        // Three-chain factorial HMM
                        using (Variable.Switch(ChainStates[0][t]))
                        {
                            using (Variable.Switch(ChainStates[1][t]))
                            {
                                using (Variable.Switch(ChainStates[2][t]))
                                {
                                    Variable<double> sumOfMeans = 
                                        EmitMean[0][ChainStates[0][t]] + 
                                        EmitMean[1][ChainStates[1][t]] + 
                                        EmitMean[2][ChainStates[2][t]];
                                    Emissions[t] = Variable.GaussianFromMeanAndPrecision(sumOfMeans, ObsPrec);
                                }
                            }
                        }
                    }
                    else
                    {
                        throw new NotImplementedException(
                            $"Factorial HMM with {NumChains} chains is not yet implemented. " +
                            "Currently supports 2-3 chains. Extending to more chains requires nested Switch blocks.");
                    }
                }
            }

            DefineInferenceEngine();
        }

        /// <summary>
        /// Defines the inference engine.
        /// </summary>
        public void DefineInferenceEngine()
        {
            Engine = new InferenceEngine(new ExpectationPropagation());
            Engine.ShowFactorGraph = false;
            Engine.ShowWarnings = true;
            Engine.ShowProgress = true;
            Engine.Compiler.WriteSourceFiles = true;
            Engine.NumberOfIterations = 75;  // More iterations for factorial model
            Engine.ShowTimings = true;
            Engine.ShowSchedule = false;
        }

        /// <summary>
        /// Initializes all chain states randomly.
        /// </summary>
        /// <param name="randomSeed">Optional random seed. If null, uses time-based seed.</param>
        public void InitialiseStatesRandomly(int? randomSeed = null)
        {
            // Reset the random number generator with a different seed each time
            if (randomSeed.HasValue)
            {
                Rand.Restart(randomSeed.Value);
            }
            else
            {
                // Use high-resolution time-based seed for true randomness across runs
                // Combine DateTime.Now.Ticks with Guid to ensure uniqueness even for rapid runs
                int seed = (int)(DateTime.Now.Ticks & 0xFFFFFFFF) ^ Guid.NewGuid().GetHashCode();
                Rand.Restart(seed);
            }
            
            for (int c = 0; c < NumChains; c++)
            {
                VariableArray<Discrete> zinit = Variable<Discrete>.Array(T);
                zinit.ObservedValue = Util.ArrayInit(T.SizeAsInt, 
                    t => Discrete.PointMass(Rand.Int(K[c].SizeAsInt), K[c].SizeAsInt));
                ChainStates[c][T].InitialiseTo(zinit[T]);
            }
        }

        /// <summary>
        /// Initializes chain states from provided assignments.
        /// </summary>
        /// <param name="assignments">Array of assignments for each chain [chain][time].</param>
        public void InitialiseStatesFromAssignments(int[][] assignments)
        {
            if (assignments.Length != NumChains)
                throw new ArgumentException($"Must provide assignments for all {NumChains} chains");

            for (int c = 0; c < NumChains; c++)
            {
                if (assignments[c].Length != T.SizeAsInt)
                    throw new ArgumentException($"Chain {c} assignments length ({assignments[c].Length}) must match time steps ({T.SizeAsInt})");

                VariableArray<Discrete> zinit = Variable<Discrete>.Array(T);
                zinit.ObservedValue = Util.ArrayInit(T.SizeAsInt, 
                    t => Discrete.PointMass(assignments[c][t], K[c].SizeAsInt));
                ChainStates[c][T].InitialiseTo(zinit[T]);
            }
        }

        /// <summary>
        /// Observes the emission data.
        /// </summary>
        /// <param name="emitData">Observed emissions.</param>
        public void ObserveData(double[] emitData)
        {
            EmitData = emitData;
            Emissions.ObservedValue = EmitData;
        }

        /// <summary>
        /// Sets priors for all chains.
        /// </summary>
        /// <param name="probInitPriors">Initial state priors for each chain.</param>
        /// <param name="cptTransPriors">Transition priors for each chain.</param>
        /// <param name="emitMeanPriors">Emission mean priors for each chain.</param>
        /// <param name="emitPrecPriors">Emission precision priors for each chain.</param>
        /// <param name="obsPrecPrior">Observation precision prior.</param>
        public void SetPriors(
            Dirichlet[] probInitPriors,
            Dirichlet[][] cptTransPriors,
            Gaussian[][] emitMeanPriors,
            Gamma[][] emitPrecPriors,
            Gamma obsPrecPrior)
        {
            if (probInitPriors.Length != NumChains)
                throw new ArgumentException($"Must provide initial priors for all {NumChains} chains");

            for (int c = 0; c < NumChains; c++)
            {
                ProbInitPrior[c].ObservedValue = probInitPriors[c];
                CPTTransPrior[c].ObservedValue = cptTransPriors[c];
                EmitMeanPrior[c].ObservedValue = emitMeanPriors[c];
                EmitPrecPrior[c].ObservedValue = emitPrecPriors[c];
            }

            ObsPrecPrior.ObservedValue = obsPrecPrior;
        }

        /// <summary>
        /// Sets uninformed priors for all chains.
        /// </summary>
        public void SetUninformedPriors()
        {
            for (int c = 0; c < NumChains; c++)
            {
                int numStates = K[c].SizeAsInt;
                ProbInitPrior[c].ObservedValue = Dirichlet.Uniform(numStates);
                CPTTransPrior[c].ObservedValue = Util.ArrayInit(numStates, 
                    k => Dirichlet.Uniform(numStates)).ToArray();
                EmitMeanPrior[c].ObservedValue = Util.ArrayInit(numStates, 
                    k => Gaussian.FromMeanAndVariance(0, 1000)).ToArray();
                EmitPrecPrior[c].ObservedValue = Util.ArrayInit(numStates, 
                    k => Gamma.FromMeanAndVariance(1, 100)).ToArray();
            }

            // Observation noise prior
            ObsPrecPrior.ObservedValue = Gamma.FromMeanAndVariance(1, 100);
        }

        /// <summary>
        /// Infers the posteriors for all chains.
        /// </summary>
        public void InferPosteriors()
        {
            // Infer posteriors for each chain
            ProbInitPosterior = new Dirichlet[NumChains];
            CPTTransPosterior = new Dirichlet[NumChains][];
            EmitMeanPosterior = new Gaussian[NumChains][];
            EmitPrecPosterior = new Gamma[NumChains][];
            ChainStatesPosterior = new Discrete[NumChains][];

            for (int c = 0; c < NumChains; c++)
            {
                ProbInitPosterior[c] = Engine.Infer<Dirichlet>(ProbInit[c]);
                CPTTransPosterior[c] = Engine.Infer<Dirichlet[]>(CPTTrans[c]);
                EmitMeanPosterior[c] = Engine.Infer<Gaussian[]>(EmitMean[c]);
                EmitPrecPosterior[c] = Engine.Infer<Gamma[]>(EmitPrec[c]);
                ChainStatesPosterior[c] = Engine.Infer<Discrete[]>(ChainStates[c]);
            }

            // Infer observation precision
            ObsPrecPosterior = Engine.Infer<Gamma>(ObsPrec);
            ModelEvidencePosterior = Engine.Infer<Bernoulli>(ModelEvidence);
        }

        /// <summary>
        /// Gets the most likely state sequence for each chain using LOCAL (marginal) decoding.
        /// WARNING: This method picks the most likely state at each time step independently,
        /// without considering temporal dependencies. For globally optimal paths, use GetViterbiStates().
        /// </summary>
        /// <returns>Array of state sequences [chain][time].</returns>
        public int[][] GetMAPStates()
        {
            int[][] mapStates = new int[NumChains][];
            for (int c = 0; c < NumChains; c++)
            {
                mapStates[c] = ChainStatesPosterior[c].Select(s => s.GetMode()).ToArray();
            }
            return mapStates;
        }

        /// <summary>
        /// Computes the globally optimal state sequence using the Viterbi algorithm.
        /// This finds the most probable joint path through all states, considering temporal dependencies.
        /// For Factorial HMM, we need to handle the interaction between chains through observations.
        /// </summary>
        /// <returns>Array of state sequences [chain][time].</returns>
        public int[][] GetViterbiStates()
        {
            int T_size = T.SizeAsInt;
            int[][] viterbiStates = new int[NumChains][];
            
            // Get inferred parameters (means) for emission distributions
            double[][] means = GetEmissionMeans();
            double obsPrec = ObsPrecPosterior.GetMean();
            double obsVar = 1.0 / obsPrec;
            
            // For factorial HMM with additive observations:
            // y_t = sum_c(mu_{z^c_t}) + noise
            // 
            // The Viterbi algorithm for factorial HMM requires considering all combinations
            // of states across chains. However, this is exponential in the number of chains.
            // 
            // Instead, we use an approximation: run Viterbi for each chain independently,
            // where for chain c, we marginalize over other chains' contributions using
            // the current posterior estimates.
            
            for (int c = 0; c < NumChains; c++)
            {
                int K_c = K[c].SizeAsInt;
                
                // Get transition probabilities for this chain
                double[][] transProbs = new double[K_c][];
                for (int i = 0; i < K_c; i++)
                {
                    Vector trans = CPTTransPosterior[c][i].GetMean();
                    transProbs[i] = new double[K_c];
                    for (int j = 0; j < K_c; j++)
                    {
                        transProbs[i][j] = trans[j];
                    }
                }
                
                // Get initial state probabilities
                Vector initProbs = ProbInitPosterior[c].GetMean();
                
                // Viterbi matrices
                double[,] delta = new double[T_size, K_c];  // Max probability at time t in state k
                int[,] psi = new int[T_size, K_c];          // Backpointer for path
                
                // For each time step, compute expected contribution from OTHER chains
                double[] otherChainContrib = new double[T_size];
                for (int t = 0; t < T_size; t++)
                {
                    double contrib = 0;
                    for (int c2 = 0; c2 < NumChains; c2++)
                    {
                        if (c2 != c)
                        {
                            // Use the marginal posterior to estimate contribution
                            Discrete stateDist = ChainStatesPosterior[c2][t];
                            Vector probs = stateDist.GetProbs();
                            for (int k = 0; k < K[c2].SizeAsInt; k++)
                            {
                                contrib += probs[k] * means[c2][k];
                            }
                        }
                    }
                    otherChainContrib[t] = contrib;
                }
                
                // Initialization (t=0)
                for (int k = 0; k < K_c; k++)
                {
                    double emissionLogProb = LogGaussianPDF(EmitData[0], means[c][k] + otherChainContrib[0], obsVar);
                    delta[0, k] = Math.Log(initProbs[k]) + emissionLogProb;
                    psi[0, k] = 0;
                }
                
                // Recursion (t=1 to T-1)
                for (int t = 1; t < T_size; t++)
                {
                    for (int k = 0; k < K_c; k++)
                    {
                        double maxProb = double.NegativeInfinity;
                        int maxState = 0;
                        
                        for (int j = 0; j < K_c; j++)
                        {
                            double prob = delta[t - 1, j] + Math.Log(transProbs[j][k]);
                            if (prob > maxProb)
                            {
                                maxProb = prob;
                                maxState = j;
                            }
                        }
                        
                        double emissionLogProb = LogGaussianPDF(EmitData[t], means[c][k] + otherChainContrib[t], obsVar);
                        delta[t, k] = maxProb + emissionLogProb;
                        psi[t, k] = maxState;
                    }
                }
                
                // Termination: find best final state
                double maxFinalProb = double.NegativeInfinity;
                int bestFinalState = 0;
                for (int k = 0; k < K_c; k++)
                {
                    if (delta[T_size - 1, k] > maxFinalProb)
                    {
                        maxFinalProb = delta[T_size - 1, k];
                        bestFinalState = k;
                    }
                }
                
                // Backtracking
                viterbiStates[c] = new int[T_size];
                viterbiStates[c][T_size - 1] = bestFinalState;
                for (int t = T_size - 2; t >= 0; t--)
                {
                    viterbiStates[c][t] = psi[t + 1, viterbiStates[c][t + 1]];
                }
            }
            
            return viterbiStates;
        }
        
        /// <summary>
        /// Computes the log of the Gaussian probability density function.
        /// </summary>
        private double LogGaussianPDF(double x, double mean, double variance)
        {
            double diff = x - mean;
            return -0.5 * Math.Log(2 * Math.PI * variance) - (diff * diff) / (2 * variance);
        }

        /// <summary>
        /// Prints the posteriors for all chains.
        /// </summary>
        public void PrintPosteriors()
        {
            for (int c = 0; c < NumChains; c++)
            {
                Console.WriteLine($"Chain {c}:");
                Console.WriteLine($"  Init: {ProbInitPosterior[c]}");
                
                for (int i = 0; i < K[c].SizeAsInt; i++)
                {
                    Console.WriteLine($"  Trans[{i}]: {CPTTransPosterior[c][i]}");
                }
                
                for (int i = 0; i < K[c].SizeAsInt; i++)
                {
                    Console.WriteLine($"  EmitMean[{i}]: {EmitMeanPosterior[c][i]}");
                }
                
                for (int i = 0; i < K[c].SizeAsInt; i++)
                {
                    Console.WriteLine($"  EmitPrec[{i}]: {EmitPrecPosterior[c][i]}");
                }
            }
            
            Console.WriteLine($"Observation Precision: {ObsPrecPosterior}");
        }

        /// <summary>
        /// Gets the number of chains in the model.
        /// </summary>
        public int GetNumChains()
        {
            return NumChains;
        }

        /// <summary>
        /// Gets the number of states for a specific chain.
        /// </summary>
        /// <param name="chainIndex">Index of the chain.</param>
        /// <returns>Number of states in that chain.</returns>
        public int GetNumStates(int chainIndex)
        {
            if (chainIndex < 0 || chainIndex >= NumChains)
                throw new ArgumentException($"Chain index must be between 0 and {NumChains - 1}");
            
            return K[chainIndex].SizeAsInt;
        }

        /// <summary>
        /// Resets the inference.
        /// </summary>
        public void ResetInference()
        {
            for (int i = 0; i < T.SizeAsInt; i++)
            {
                double emit = Emissions[i].ObservedValue;
                Emissions[i].ClearObservedValue();
                Emissions[i].ObservedValue = emit;
            }
        }

        /// <summary>
        /// Gets the inferred emission contributions from each chain.
        /// </summary>
        /// <returns>Array of emission means [chain][state].</returns>
        public double[][] GetEmissionMeans()
        {
            double[][] means = new double[NumChains][];
            for (int c = 0; c < NumChains; c++)
            {
                means[c] = EmitMeanPosterior[c].Select(g => g.GetMean()).ToArray();
            }
            return means;
        }

        /// <summary>
        /// Gets the state transition matrices for all chains.
        /// </summary>
        /// <returns>Array of transition matrices [chain][from_state, to_state].</returns>
        public double[][][] GetTransitionMatrices()
        {
            double[][][] transitions = new double[NumChains][][];
            for (int c = 0; c < NumChains; c++)
            {
                int numStates = K[c].SizeAsInt;
                transitions[c] = new double[numStates][];
                
                for (int from = 0; from < numStates; from++)
                {
                    transitions[c][from] = CPTTransPosterior[c][from].GetMean().ToArray();
                }
            }
            return transitions;
        }

        /// <summary>
        /// Decomposes observations into per-chain contributions.
        /// </summary>
        /// <returns>Array of estimated contributions [chain][time].</returns>
        public double[][] DecomposeObservations()
        {
            double[][] contributions = new double[NumChains][];
            int[][] mapStates = GetMAPStates();
            double[][] emissionMeans = GetEmissionMeans();

            for (int c = 0; c < NumChains; c++)
            {
                contributions[c] = new double[T.SizeAsInt];
                for (int t = 0; t < T.SizeAsInt; t++)
                {
                    int state = mapStates[c][t];
                    contributions[c][t] = emissionMeans[c][state];
                }
            }

            return contributions;
        }

        /// <summary>
        /// Returns a string representation summarizing the model.
        /// </summary>
        public override string ToString()
        {
            string output = $"Factorial HMM with {NumChains} chains\n";
            output += $"Sequence length: {T.SizeAsInt}\n";
            
            for (int c = 0; c < NumChains; c++)
            {
                output += $"Chain {c}: {K[c].SizeAsInt} states\n";
            }
            
            return output;
        }
    }
}

