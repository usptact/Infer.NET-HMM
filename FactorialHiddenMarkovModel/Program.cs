using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace FactorialHiddenMarkovModel
{
    /// <summary>
    /// Program for Factorial HMM inference.
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                ShowUsage();
                return;
            }

            // Check if in generation mode
            if (args[0].Equals("--generate", StringComparison.OrdinalIgnoreCase) || 
                args[0].Equals("-g", StringComparison.OrdinalIgnoreCase))
            {
                RunGenerateMode(args);
                return;
            }

            // Inference mode
            string csvFilePath = args[0];
            
            if (args.Length < 2)
            {
                Console.WriteLine("Error: Must specify number of chains");
                ShowUsage();
                return;
            }

            if (!int.TryParse(args[1], out int numChains) || numChains < 2 || numChains > 3)
            {
                if (numChains == 1)
                {
                    Console.WriteLine("Error: For single-chain HMM, please use the HiddenMarkovModel program instead.");
                    Console.WriteLine("Factorial HMM is designed for 2 or more independent chains.");
                    return;
                }
                Console.WriteLine("Error: Number of chains must be between 2 and 3");
                return;
            }

            // Parse number of states for each chain
            int[] numStatesPerChain = new int[numChains];
            
            if (args.Length < 2 + numChains)
            {
                Console.WriteLine($"Error: Must specify number of states for each of the {numChains} chains");
                ShowUsage();
                return;
            }

            for (int c = 0; c < numChains; c++)
            {
                if (!int.TryParse(args[2 + c], out numStatesPerChain[c]) || numStatesPerChain[c] < 2)
                {
                    Console.WriteLine($"Error: Number of states for chain {c} must be >= 2");
                    return;
                }
            }

            // Validate file exists
            if (!File.Exists(csvFilePath))
            {
                Console.WriteLine($"Error: File '{csvFilePath}' not found.");
                return;
            }

            try
            {
                // Load data
                double[] observations = LoadDataFromCsv(csvFilePath);
                
                if (observations.Length == 0)
                {
                    Console.WriteLine("Error: No data found in CSV file.");
                    return;
                }

                Console.WriteLine($"Loaded {observations.Length} observations from '{csvFilePath}'");
                Console.WriteLine($"Number of chains: {numChains}");
                for (int c = 0; c < numChains; c++)
                {
                    Console.WriteLine($"  Chain {c}: {numStatesPerChain[c]} states");
                }
                Console.WriteLine();

                // Run Factorial HMM inference
                RunFactorialHMMInference(observations, numStatesPerChain, csvFilePath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"  {ex.InnerException.Message}");
                }
            }
        }

        static void ShowUsage()
        {
            Console.WriteLine("Factorial Hidden Markov Model - Infer.NET Implementation");
            Console.WriteLine("=========================================================");
            Console.WriteLine();
            Console.WriteLine("INFERENCE MODE:");
            Console.WriteLine("  FactorialHiddenMarkovModel <csv_file> <num_chains> <states_chain_0> [states_chain_1] [states_chain_2]");
            Console.WriteLine();
            Console.WriteLine("  Arguments:");
            Console.WriteLine("    csv_file        Path to CSV file containing observation data");
            Console.WriteLine("    num_chains      Number of independent chains (2-3)");
            Console.WriteLine("    states_chain_i  Number of states for chain i (>= 2)");
            Console.WriteLine();
            Console.WriteLine("  Examples:");
            Console.WriteLine("    FactorialHiddenMarkovModel data.csv 2 2 2");
            Console.WriteLine("    FactorialHiddenMarkovModel data.csv 3 2 3 2");
            Console.WriteLine();
            Console.WriteLine("GENERATION MODE:");
            Console.WriteLine("  FactorialHiddenMarkovModel --generate <output_file> <length> <num_chains> <states_per_chain...> [chain_params...]");
            Console.WriteLine("  FactorialHiddenMarkovModel -g <output_file> <length> <num_chains> <states_per_chain...> [chain_params...]");
            Console.WriteLine();
            Console.WriteLine("  Arguments:");
            Console.WriteLine("    output_file     Path to output CSV file");
            Console.WriteLine("    length          Number of observations to generate");
            Console.WriteLine("    num_chains      Number of chains (2-3)");
            Console.WriteLine("    states_chain_i  Number of states for each chain");
            Console.WriteLine("    chain_params    Optional: For each chain, for each state: mean,variance");
            Console.WriteLine("                    Format: chain0_state0_mean,var chain0_state1_mean,var ...");
            Console.WriteLine();
            Console.WriteLine("  Examples:");
            Console.WriteLine("    # 2 chains with random parameters");
            Console.WriteLine("    FactorialHiddenMarkovModel -g synthetic.csv 200 2 2 2");
            Console.WriteLine();
            Console.WriteLine("    # 2 chains with specified parameters");
            Console.WriteLine("    FactorialHiddenMarkovModel -g synthetic.csv 200 2 2 2 -5,2 5,2 -3,1 3,1");
            Console.WriteLine();
            Console.WriteLine("Note: For single-chain HMM, use the HiddenMarkovModel program instead.");
        }

        static double[] LoadDataFromCsv(string filePath)
        {
            var data = new List<double>();
            int lineNumber = 0;

            foreach (string line in File.ReadLines(filePath))
            {
                lineNumber++;
                string trimmedLine = line.Trim();

                if (string.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("#"))
                {
                    continue;
                }

                if (double.TryParse(trimmedLine, out double value))
                {
                    data.Add(value);
                }
                else
                {
                    Console.WriteLine($"Warning: Skipping invalid data at line {lineNumber}: '{trimmedLine}'");
                }
            }

            return data.ToArray();
        }

        static void RunFactorialHMMInference(double[] observations, int[] numStatesPerChain, string inputFileName)
        {
            int T = observations.Length;
            int numChains = numStatesPerChain.Length;

            Console.WriteLine("Setting up Factorial HMM with data-informed priors...");
            
            // Calculate data statistics
            double dataMean = observations.Average();
            double dataVariance = observations.Select(x => Math.Pow(x - dataMean, 2)).Average();
            double dataStdDev = Math.Sqrt(dataVariance);
            double dataMin = observations.Min();
            double dataMax = observations.Max();
            double dataRange = dataMax - dataMin;

            Console.WriteLine($"Data statistics:");
            Console.WriteLine($"  Mean: {dataMean:F4}");
            Console.WriteLine($"  Std Dev: {dataStdDev:F4}");
            Console.WriteLine($"  Min: {dataMin:F4}");
            Console.WriteLine($"  Max: {dataMax:F4}");
            Console.WriteLine($"  Range: {dataRange:F4}");
            Console.WriteLine();

            // Set up priors for each chain
            Dirichlet[] initPriors = new Dirichlet[numChains];
            Dirichlet[][] transPriors = new Dirichlet[numChains][];
            Gaussian[][] meanPriors = new Gaussian[numChains][];
            Gamma[][] precPriors = new Gamma[numChains][];

            // Create random number generator for prior initialization
            Random rand = new Random();
            
            for (int c = 0; c < numChains; c++)
            {
                int K = numStatesPerChain[c];
                
                // Slightly randomized priors to break symmetry
                // Use random but not too extreme (between 0.5 and 2.0)
                double[] initCounts = new double[K];
                for (int k = 0; k < K; k++)
                {
                    initCounts[k] = 0.5 + rand.NextDouble() * 1.5; // Range: [0.5, 2.0]
                }
                initPriors[c] = new Dirichlet(Vector.FromArray(initCounts));
                
                // Also randomize transition priors slightly
                transPriors[c] = new Dirichlet[K];
                for (int k = 0; k < K; k++)
                {
                    double[] transCounts = new double[K];
                    for (int k2 = 0; k2 < K; k2++)
                    {
                        transCounts[k2] = 0.5 + rand.NextDouble() * 1.5;
                    }
                    transPriors[c][k] = new Dirichlet(Vector.FromArray(transCounts));
                }

                // Spread mean priors across a portion of the data range
                // Each chain gets a fraction of the total range
                meanPriors[c] = new Gaussian[K];
                double chainRangePerState = dataRange / (numChains * K);
                
                for (int k = 0; k < K; k++)
                {
                    // Distribute priors symmetrically around zero
                    double priorMean = -dataRange / (2.0 * numChains) + (k + 0.5) * chainRangePerState;
                    double priorVariance = dataVariance * 20; // Weak prior
                    meanPriors[c][k] = Gaussian.FromMeanAndVariance(priorMean, priorVariance);
                }

                // Weakly informed precision priors
                precPriors[c] = Enumerable.Repeat(
                    Gamma.FromMeanAndVariance(1.0 / (dataVariance / numChains), 100), K
                ).ToArray();
            }

            // Observation noise prior
            Gamma obsPrecPrior = Gamma.FromMeanAndVariance(1.0 / (dataVariance / 10), 100);

            Console.WriteLine("Prior mean ranges for each chain:");
            for (int c = 0; c < numChains; c++)
            {
                double minPrior = meanPriors[c].Min(g => g.GetMean());
                double maxPrior = meanPriors[c].Max(g => g.GetMean());
                Console.WriteLine($"  Chain {c}: [{minPrior:F4}, {maxPrior:F4}]");
            }
            Console.WriteLine();

            // Create and configure Factorial HMM
            Console.WriteLine("Building Factorial HMM model...");
            FactorialHiddenMarkovModel model = new FactorialHiddenMarkovModel(T, numStatesPerChain);
            model.SetPriors(initPriors, transPriors, meanPriors, precPriors, obsPrecPrior);
            model.ObserveData(observations);
            model.InitialiseStatesRandomly();

            Console.WriteLine("Running inference...");
            model.InferPosteriors();

            // Get results using Viterbi algorithm for globally optimal state sequences
            Console.WriteLine("Computing globally optimal state sequences using Viterbi algorithm...");
            int[][] inferredStates = model.GetViterbiStates();
            double[][] emissionMeans = model.GetEmissionMeans();
            double[][][] transitionMatrices = model.GetTransitionMatrices();
            
            // Optionally compare with local (marginal) decoding
            if (false) // Set to true to see comparison
            {
                int[][] localStates = model.GetMAPStates();
                Console.WriteLine();
                Console.WriteLine("Comparing Viterbi (global) vs Marginal (local) decoding:");
                for (int c = 0; c < numChains; c++)
                {
                    int differences = 0;
                    for (int t = 0; t < observations.Length; t++)
                    {
                        if (inferredStates[c][t] != localStates[c][t])
                            differences++;
                    }
                    double diffPercent = (differences / (double)observations.Length) * 100;
                    Console.WriteLine($"  Chain {c}: {differences} differences ({diffPercent:F1}% of time steps)");
                }
                Console.WriteLine();
            }

            // Print results
            Console.WriteLine();
            Console.WriteLine("=== RESULTS ===");
            Console.WriteLine();

            // Display model evidence
            double logEvidence = model.ModelEvidencePosterior.LogOdds;
            Console.WriteLine($"Model Evidence (log): {logEvidence:F2}");
            Console.WriteLine();

            Console.WriteLine("Inferred emission means by chain:");
            for (int c = 0; c < numChains; c++)
            {
                Console.WriteLine($"Chain {c}:");
                for (int k = 0; k < numStatesPerChain[c]; k++)
                {
                    Console.WriteLine($"  State {k}: mean={emissionMeans[c][k]:F4}");
                }
            }
            Console.WriteLine();

            Console.WriteLine($"Observation noise precision: {model.ObsPrecPosterior.GetMean():F4} " +
                            $"(variance: {1.0 / model.ObsPrecPosterior.GetMean():F4})");
            Console.WriteLine();

            // State distributions for each chain
            for (int c = 0; c < numChains; c++)
            {
                Console.WriteLine($"Chain {c} State Distribution:");
                int[] stateCounts = new int[numStatesPerChain[c]];
                for (int t = 0; t < T; t++)
                {
                    stateCounts[inferredStates[c][t]]++;
                }
                
                for (int k = 0; k < numStatesPerChain[c]; k++)
                {
                    double percentage = (stateCounts[k] / (double)T) * 100;
                    Console.WriteLine($"  State {k}: {stateCounts[k]} occurrences ({percentage:F1}%)");
                }
                Console.WriteLine();
            }

            // Transition matrices
            for (int c = 0; c < numChains; c++)
            {
                Console.WriteLine($"Chain {c} Transition Matrix:");
                Console.WriteLine("(rows: from state, columns: to state, values: probability)");
                Console.WriteLine();

                int K = numStatesPerChain[c];
                
                // Print header
                Console.Write("        ");
                for (int to = 0; to < K; to++)
                {
                    Console.Write($"State {to}  ");
                }
                Console.WriteLine();
                
                Console.Write("        ");
                for (int to = 0; to < K; to++)
                {
                    Console.Write("--------  ");
                }
                Console.WriteLine();

                // Print matrix rows
                for (int from = 0; from < K; from++)
                {
                    Console.Write($"State {from} ");
                    for (int to = 0; to < K; to++)
                    {
                        Console.Write($"{transitionMatrices[c][from][to],8:F4}  ");
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
            }

            // Save results
            string outputFile = Path.ChangeExtension(Path.GetFileName(inputFileName), ".factorial.csv");
            SaveResults(outputFile, observations, inferredStates, numChains);
            Console.WriteLine($"Results saved to '{outputFile}'");
        }

        static void SaveResults(string filePath, double[] observations, int[][] chainStates, int numChains)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                // Write header
                writer.Write("# Factorial HMM Inference Results\n");
                writer.Write($"# Number of chains: {numChains}\n");
                writer.Write("# Columns: Observation");
                for (int c = 0; c < numChains; c++)
                {
                    writer.Write($",Chain{c}_State");
                }
                writer.Write("\n");
                
                // Write data
                for (int t = 0; t < observations.Length; t++)
                {
                    writer.Write($"{observations[t]:F4}");
                    for (int c = 0; c < numChains; c++)
                    {
                        writer.Write($",{chainStates[c][t]}");
                    }
                    writer.Write("\n");
                }
            }
        }

        static void RunGenerateMode(string[] args)
        {
            // Parse: -g <output_file> <length> <num_chains> <states_chain_0> ... [params]
            if (args.Length < 4)
            {
                Console.WriteLine("Error: Insufficient arguments for generation mode.");
                ShowUsage();
                return;
            }

            string outputFile = args[1];

            if (!int.TryParse(args[2], out int length) || length < 1)
            {
                Console.WriteLine("Error: Length must be a positive integer.");
                return;
            }

            if (!int.TryParse(args[3], out int numChains) || numChains < 2 || numChains > 3)
            {
                if (numChains == 1)
                {
                    Console.WriteLine("Error: For single-chain HMM, use the HiddenMarkovModel program instead.");
                    return;
                }
                Console.WriteLine("Error: Number of chains must be between 2 and 3");
                return;
            }

            // Parse states per chain
            if (args.Length < 4 + numChains)
            {
                Console.WriteLine($"Error: Must specify number of states for each of the {numChains} chains");
                return;
            }

            int[] numStatesPerChain = new int[numChains];
            for (int c = 0; c < numChains; c++)
            {
                if (!int.TryParse(args[4 + c], out numStatesPerChain[c]) || numStatesPerChain[c] < 2)
                {
                    Console.WriteLine($"Error: Number of states for chain {c} must be >= 2");
                    return;
                }
            }

            int totalStates = numStatesPerChain.Sum();

            // Parse chain parameters if provided
            double[][] means = new double[numChains][];
            double[][] variances = new double[numChains][];
            Random rand = new Random();

            // Check if user provided parameters
            int expectedParams = totalStates;
            bool hasParams = args.Length >= 4 + numChains + expectedParams;

            if (hasParams)
            {
                // Parse user-provided parameters
                try
                {
                    int paramIndex = 4 + numChains;
                    for (int c = 0; c < numChains; c++)
                    {
                        int K = numStatesPerChain[c];
                        means[c] = new double[K];
                        variances[c] = new double[K];

                        for (int k = 0; k < K; k++)
                        {
                            string[] parts = args[paramIndex++].Split(',');
                            if (parts.Length != 2)
                            {
                                Console.WriteLine($"Error: Chain {c} State {k} parameter must be 'mean,variance'");
                                return;
                            }

                            if (!double.TryParse(parts[0], out means[c][k]))
                            {
                                Console.WriteLine($"Error: Invalid mean for chain {c} state {k}: '{parts[0]}'");
                                return;
                            }

                            if (!double.TryParse(parts[1], out variances[c][k]) || variances[c][k] <= 0)
                            {
                                Console.WriteLine($"Error: Invalid variance for chain {c} state {k}: '{parts[1]}'");
                                return;
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error parsing parameters: {ex.Message}");
                    return;
                }
            }
            else
            {
                // Generate random parameters
                Console.WriteLine("No parameters provided. Generating random parameters...");
                
                for (int c = 0; c < numChains; c++)
                {
                    int K = numStatesPerChain[c];
                    means[c] = new double[K];
                    variances[c] = new double[K];

                    // Generate well-separated means for this chain
                    int baseMean = rand.Next(-20, 21);
                    double separation = 5.0 + rand.NextDouble() * 10.0; // 5-15 units apart

                    for (int k = 0; k < K; k++)
                    {
                        means[c][k] = baseMean + k * separation;
                        variances[c][k] = 1.0 + rand.NextDouble() * 3.0; // Variance 1-4
                    }
                }
            }

            // Display generation parameters
            Console.WriteLine("Generating synthetic Factorial HMM sequence...");
            Console.WriteLine($"  Output file: {outputFile}");
            Console.WriteLine($"  Length: {length}");
            Console.WriteLine($"  Number of chains: {numChains}");
            Console.WriteLine();
            
            Console.WriteLine("Chain parameters:");
            for (int c = 0; c < numChains; c++)
            {
                Console.WriteLine($"Chain {c}: {numStatesPerChain[c]} states");
                for (int k = 0; k < numStatesPerChain[c]; k++)
                {
                    Console.WriteLine($"  State {k}: mean={means[c][k]:F2}, variance={variances[c][k]:F2}");
                }
            }
            Console.WriteLine();

            try
            {
                var (observations, chainStates, transitionMatrices, initialProbs) = 
                    GenerateFactorialHMMSequence(length, numStatesPerChain, means, variances, rand);

                SaveGeneratedSequence(outputFile, observations, chainStates, means, variances, 
                                    transitionMatrices, initialProbs);

                // Display statistics
                Console.WriteLine("Generation complete!");
                Console.WriteLine();
                
                for (int c = 0; c < numChains; c++)
                {
                    Console.WriteLine($"Chain {c} state distribution:");
                    var stateCounts = new int[numStatesPerChain[c]];
                    for (int t = 0; t < length; t++)
                    {
                        stateCounts[chainStates[c][t]]++;
                    }
                    
                    for (int k = 0; k < numStatesPerChain[c]; k++)
                    {
                        double percentage = (stateCounts[k] / (double)length) * 100;
                        Console.WriteLine($"  State {k}: {stateCounts[k]} occurrences ({percentage:F1}%)");
                    }
                }
                
                Console.WriteLine();
                Console.WriteLine($"Data saved to '{outputFile}'");
                Console.WriteLine($"True states saved to '{Path.ChangeExtension(outputFile, ".chains.csv")}'");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating sequence: {ex.Message}");
            }
        }

        static (double[] observations, int[][] chainStates, double[][][]transitionMatrices, double[][] initialProbs)
            GenerateFactorialHMMSequence(int length, int[] numStatesPerChain, double[][] means, 
                                        double[][] variances, Random rand)
        {
            int numChains = numStatesPerChain.Length;
            double[] observations = new double[length];
            int[][] chainStates = new int[numChains][];
            double[][][] transitionMatrices = new double[numChains][][];
            double[][] initialProbs = new double[numChains][];

            // Generate parameters for each chain
            for (int c = 0; c < numChains; c++)
            {
                int K = numStatesPerChain[c];
                chainStates[c] = new int[length];

                // Generate transition matrix
                transitionMatrices[c] = new double[K][];
                for (int from = 0; from < K; from++)
                {
                    transitionMatrices[c][from] = new double[K];
                    double sum = 0;
                    for (int to = 0; to < K; to++)
                    {
                        transitionMatrices[c][from][to] = rand.NextDouble();
                        sum += transitionMatrices[c][from][to];
                    }
                    // Normalize
                    for (int to = 0; to < K; to++)
                    {
                        transitionMatrices[c][from][to] /= sum;
                    }
                }

                // Generate initial probabilities
                initialProbs[c] = new double[K];
                double initSum = 0;
                for (int k = 0; k < K; k++)
                {
                    initialProbs[c][k] = rand.NextDouble();
                    initSum += initialProbs[c][k];
                }
                for (int k = 0; k < K; k++)
                {
                    initialProbs[c][k] /= initSum;
                }
            }

            // Generate sequence
            double obsNoise = 0.1; // Small observation noise

            for (int t = 0; t < length; t++)
            {
                double sumOfEmissions = 0;

                for (int c = 0; c < numChains; c++)
                {
                    // Sample state for this chain
                    if (t == 0)
                    {
                        chainStates[c][t] = SampleFromDiscrete(initialProbs[c], rand);
                    }
                    else
                    {
                        int prevState = chainStates[c][t - 1];
                        chainStates[c][t] = SampleFromDiscrete(transitionMatrices[c][prevState], rand);
                    }

                    // Add emission from this chain
                    int state = chainStates[c][t];
                    double emission = SampleFromGaussian(means[c][state], Math.Sqrt(variances[c][state]), rand);
                    sumOfEmissions += emission;
                }

                // Add observation noise
                observations[t] = sumOfEmissions + SampleFromGaussian(0, Math.Sqrt(obsNoise), rand);
            }

            return (observations, chainStates, transitionMatrices, initialProbs);
        }

        static int SampleFromDiscrete(double[] probabilities, Random rand)
        {
            double u = rand.NextDouble();
            double cumulative = 0;
            
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (u <= cumulative)
                {
                    return i;
                }
            }
            
            return probabilities.Length - 1;
        }

        static double SampleFromGaussian(double mean, double stdDev, Random rand)
        {
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return mean + stdDev * z0;
        }

        static void SaveGeneratedSequence(string filePath, double[] observations, int[][] chainStates,
            double[][] means, double[][] variances, double[][][] transitionMatrices, double[][] initialProbs)
        {
            int numChains = chainStates.Length;
            
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("# Synthetic Factorial HMM Sequence");
                writer.WriteLine($"# Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                writer.WriteLine($"# Length: {observations.Length}");
                writer.WriteLine($"# Number of chains: {numChains}");
                writer.WriteLine("#");
                
                // Chain parameters
                for (int c = 0; c < numChains; c++)
                {
                    writer.WriteLine($"# Chain {c} ({means[c].Length} states):");
                    for (int k = 0; k < means[c].Length; k++)
                    {
                        writer.WriteLine($"#   State {k}: mean={means[c][k]:F2}, variance={variances[c][k]:F2}");
                    }
                    writer.WriteLine("#");
                }

                // Initial probabilities
                for (int c = 0; c < numChains; c++)
                {
                    writer.WriteLine($"# Chain {c} initial probabilities:");
                    for (int k = 0; k < initialProbs[c].Length; k++)
                    {
                        writer.WriteLine($"#   P(State {k}) = {initialProbs[c][k]:F4}");
                    }
                    writer.WriteLine("#");
                }

                // Transition matrices
                for (int c = 0; c < numChains; c++)
                {
                    writer.WriteLine($"# Chain {c} transition matrix:");
                    writer.WriteLine("# (rows: from state, columns: to state, values: probability)");
                    writer.WriteLine("#");
                    
                    int K = means[c].Length;
                    
                    // Header
                    writer.Write("#         ");
                    for (int to = 0; to < K; to++)
                    {
                        writer.Write($"State {to}  ");
                    }
                    writer.WriteLine();
                    
                    writer.Write("#         ");
                    for (int to = 0; to < K; to++)
                    {
                        writer.Write("--------  ");
                    }
                    writer.WriteLine();
                    
                    // Matrix rows
                    for (int from = 0; from < K; from++)
                    {
                        writer.Write($"# State {from} ");
                        for (int to = 0; to < K; to++)
                        {
                            writer.Write($"{transitionMatrices[c][from][to],8:F4}  ");
                        }
                        writer.WriteLine();
                    }
                    writer.WriteLine("#");
                }

                writer.WriteLine("# Format: observation");
                writer.WriteLine("#");
                
                // Write observations
                for (int i = 0; i < observations.Length; i++)
                {
                    writer.WriteLine($"{observations[i]:F4}");
                }
            }

            // Save true chain states to separate file
            string statesFile = Path.ChangeExtension(filePath, ".chains.csv");
            using (StreamWriter writer = new StreamWriter(statesFile))
            {
                writer.WriteLine("# True hidden states for each chain");
                writer.Write("# Index");
                for (int c = 0; c < numChains; c++)
                {
                    writer.Write($",Chain{c}_State");
                }
                writer.WriteLine();
                
                for (int t = 0; t < observations.Length; t++)
                {
                    writer.Write($"{t}");
                    for (int c = 0; c < numChains; c++)
                    {
                        writer.Write($",{chainStates[c][t]}");
                    }
                    writer.WriteLine();
                }
            }
        }
    }
}


