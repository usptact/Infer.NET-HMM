using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic;
using System.IO;

namespace HiddenMarkovModel
{
    /// <summary>
    /// Program.
    /// </summary>
    class Program
    {
        /// <summary>
        /// The entry point of the program, where the program control starts and ends.
        /// </summary>
        /// <param name="args">The command-line arguments.</param>
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
            int numStates = 2; // Default number of states

            // Check for optional number of states parameter
            if (args.Length > 1)
            {
                if (!int.TryParse(args[1], out numStates) || numStates < 2)
                {
                    Console.WriteLine("Error: Number of states must be an integer >= 2");
                    ShowUsage();
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
                // Load data from CSV
                double[] emissions = LoadDataFromCsv(csvFilePath);
                
                if (emissions.Length == 0)
                {
                    Console.WriteLine("Error: No data found in CSV file.");
                    return;
                }

                Console.WriteLine($"Loaded {emissions.Length} observations from '{csvFilePath}'");
                Console.WriteLine($"Using {numStates} hidden states");
                Console.WriteLine();

                // Run HMM inference
                RunHMMInference(emissions, numStates, csvFilePath);
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

        /// <summary>
        /// Shows usage information.
        /// </summary>
        static void ShowUsage()
        {
            Console.WriteLine("Hidden Markov Model - Infer.NET Implementation");
            Console.WriteLine("===============================================");
            Console.WriteLine();
            Console.WriteLine("INFERENCE MODE:");
            Console.WriteLine("  HiddenMarkovModel <csv_file> [num_states]");
            Console.WriteLine();
            Console.WriteLine("  Arguments:");
            Console.WriteLine("    csv_file    Path to CSV file containing observation data");
            Console.WriteLine("                (one numeric value per line)");
            Console.WriteLine("    num_states  Optional. Number of hidden states (default: 2)");
            Console.WriteLine();
            Console.WriteLine("  Examples:");
            Console.WriteLine("    HiddenMarkovModel data.csv");
            Console.WriteLine("    HiddenMarkovModel data.csv 3");
            Console.WriteLine();
            Console.WriteLine("GENERATION MODE:");
            Console.WriteLine("  HiddenMarkovModel --generate <output_file> <length> <num_states> [state_params...]");
            Console.WriteLine("  HiddenMarkovModel -g <output_file> <length> <num_states> [state_params...]");
            Console.WriteLine();
            Console.WriteLine("  Arguments:");
            Console.WriteLine("    output_file   Path to output CSV file");
            Console.WriteLine("    length        Number of observations to generate");
            Console.WriteLine("    num_states    Number of hidden states (>= 2)");
            Console.WriteLine("    state_params  Optional. For each state: mean,variance");
            Console.WriteLine("                  If not provided, random means with small variance");
            Console.WriteLine();
            Console.WriteLine("  Examples:");
            Console.WriteLine("    # Generate 100 observations with 2 states (random parameters)");
            Console.WriteLine("    HiddenMarkovModel --generate synthetic.csv 100 2");
            Console.WriteLine();
            Console.WriteLine("    # Generate with specified parameters for 2 states");
            Console.WriteLine("    HiddenMarkovModel -g synthetic.csv 100 2 -20,5 -30,5");
            Console.WriteLine();
            Console.WriteLine("    # Generate with 3 states");
            Console.WriteLine("    HiddenMarkovModel -g synthetic.csv 100 3 -20,5 -30,5 -10,3");
            Console.WriteLine();
            Console.WriteLine("CSV Format:");
            Console.WriteLine("  The CSV file contains one numeric observation per line.");
            Console.WriteLine("  Comments (lines starting with #) and empty lines are ignored.");
        }

        /// <summary>
        /// Robust initialization that ensures all states get some representation.
        /// Uses a hybrid approach: K-means for most points, but forces diversity.
        /// </summary>
        /// <param name="data">The observation data.</param>
        /// <param name="K">Number of states.</param>
        /// <param name="centroids">Centroids from K-means.</param>
        /// <returns>Balanced state assignments.</returns>
        static int[] RobustInitialization(double[] data, int K, double[] centroids)
        {
            int N = data.Length;
            int[] assignments = new int[N];
            int minPointsPerState = Math.Max(1, N / (K * 5)); // At least 20% of equal share
            
            // Sort centroids to get ordering
            var sortedCentroids = centroids.Select((c, i) => new { Centroid = c, Index = i })
                                          .OrderBy(x => x.Centroid)
                                          .ToArray();
            
            // Create data points with indices
            var dataWithIndices = data.Select((value, index) => new { Value = value, Index = index })
                                     .OrderBy(x => x.Value)
                                     .ToArray();
            
            // Strategy 1: Use quantile-based assignment to guarantee coverage
            // Divide the sorted data into K equal-sized regions
            for (int i = 0; i < N; i++)
            {
                // Determine which quantile region this point falls into
                int quantileRegion = Math.Min(K - 1, i * K / N);
                
                // Assign to the state corresponding to this quantile
                assignments[dataWithIndices[i].Index] = sortedCentroids[quantileRegion].Index;
            }
            
            // Strategy 2: Refine by moving points closer to actual centroids
            // while maintaining minimum representation for each state
            for (int iter = 0; iter < 5; iter++)
            {
                int[] currentCounts = new int[K];
                foreach (int a in assignments)
                    currentCounts[a]++;
                
                // For each point, try to assign it to nearest centroid
                // but only if it doesn't violate minimum counts
                for (int i = 0; i < N; i++)
                {
                    int currentAssignment = assignments[i];
                    int nearestState = 0;
                    double minDist = Math.Abs(data[i] - centroids[0]);
                    
                    for (int k = 1; k < K; k++)
                    {
                        double dist = Math.Abs(data[i] - centroids[k]);
                        if (dist < minDist)
                        {
                            minDist = dist;
                            nearestState = k;
                        }
                    }
                    
                    // Only reassign if:
                    // 1. It moves to a closer centroid
                    // 2. Current state won't drop below minimum
                    if (nearestState != currentAssignment &&
                        currentCounts[currentAssignment] > minPointsPerState)
                    {
                        assignments[i] = nearestState;
                        currentCounts[currentAssignment]--;
                        currentCounts[nearestState]++;
                    }
                }
            }
            
            return assignments;
        }

        /// <summary>
        /// Loads data from a CSV file.
        /// </summary>
        /// <param name="filePath">Path to the CSV file.</param>
        /// <returns>Array of observations.</returns>
        static double[] LoadDataFromCsv(string filePath)
        {
            var data = new List<double>();
            int lineNumber = 0;

            foreach (string line in File.ReadLines(filePath))
            {
                lineNumber++;
                string trimmedLine = line.Trim();

                // Skip empty lines and comments
                if (string.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("#"))
                {
                    continue;
                }

                // Try to parse as double
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

        /// <summary>
        /// Runs HMM inference on the provided data.
        /// </summary>
        /// <param name="emissions">Observation data.</param>
        /// <param name="numStates">Number of hidden states.</param>
        /// <param name="inputFileName">Input file name for output naming.</param>
        static void RunHMMInference(double[] emissions, int numStates, string inputFileName)
        {
            int T = emissions.Length;
            int K = numStates;

            Console.WriteLine("Setting up HMM with weakly informed priors...");
            
            // Set uninformed hyperparameters for transition probabilities
            Dirichlet ProbInitPriorObs = Dirichlet.Uniform(K);
            Dirichlet[] CPTTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();
            
            // Calculate data statistics
            double dataMean = emissions.Average();
            double dataVariance = emissions.Select(x => Math.Pow(x - dataMean, 2)).Average();
            double dataStdDev = Math.Sqrt(dataVariance);
            double dataMin = emissions.Min();
            double dataMax = emissions.Max();
            double dataRange = dataMax - dataMin;
            
            // Use weakly informed priors for emission distributions
            // Key: Give each state a DIFFERENT initial prior mean to break symmetry
            // Spread the prior means evenly across the data range
            Gaussian[] EmitMeanPriorObs = new Gaussian[K];
            for (int k = 0; k < K; k++)
            {
                // Spread means across the data range
                double priorMean = dataMin + (dataRange * (k + 0.5) / K);
                // Use very broad variance to allow data to dominate
                double priorVariance = dataVariance * 100;
                EmitMeanPriorObs[k] = Gaussian.FromMeanAndVariance(priorMean, priorVariance);
            }
            
            // Set weakly informed priors for emission precisions (inverse variance)
            // Use a very broad prior with high variance to allow flexibility
            Gamma[] EmitPrecPriorObs = Enumerable.Repeat(
                Gamma.FromMeanAndVariance(1.0 / dataVariance, 100.0), K
            ).ToArray();

            Console.WriteLine($"Data statistics:");
            Console.WriteLine($"  Mean: {dataMean:F4}");
            Console.WriteLine($"  Std Dev: {dataStdDev:F4}");
            Console.WriteLine($"  Min: {dataMin:F4}");
            Console.WriteLine($"  Max: {dataMax:F4}");
            Console.WriteLine($"  Range: {dataRange:F4}");
            Console.WriteLine();

            Console.WriteLine("Prior means for emission distributions:");
            for (int k = 0; k < K; k++)
            {
                Console.WriteLine($"  State {k}: mean={EmitMeanPriorObs[k].GetMean():F4}, variance={EmitMeanPriorObs[k].GetVariance():F2}");
            }
            Console.WriteLine();

            // Create and configure HMM
            Console.WriteLine("Building HMM model...");
            HiddenMarkovModel model = new HiddenMarkovModel(T, K);
            model.SetPriors(ProbInitPriorObs, CPTTransPriorObs, EmitMeanPriorObs, EmitPrecPriorObs);
            model.ObserveData(emissions);
            
            // Use K-means to get better initial state estimates
            Console.WriteLine("Initializing states using K-means clustering...");
            KMeans kmeans = new KMeans(K);
            int[] kmeansAssignments = kmeans.Fit(emissions, verbose: true);
            
            // Check if K-means resulted in balanced clusters
            int[] clusterCounts = kmeans.GetClusterCounts();
            bool hasEmptyCluster = clusterCounts.Any(c => c == 0);
            bool hasVerySmallCluster = clusterCounts.Any(c => c < T / (K * 4)); // Less than 25% of expected
            
            // Also check for very dominant clusters (one cluster has >80% of points)
            bool hasVeryDominantCluster = clusterCounts.Any(c => c > (T * 4) / 5);
            
            // Check for heavily overlapping distributions
            // If centroids are very close relative to data spread, K-means may be unreliable
            double centroidSpread = K > 1 ? kmeans.Centroids.Max() - kmeans.Centroids.Min() : dataRange;
            double overlapRatio = centroidSpread / dataRange;
            bool hasHeavyOverlap = overlapRatio < 0.3; // Centroids span < 30% of data range
            
            if (hasEmptyCluster || hasVerySmallCluster || hasVeryDominantCluster || hasHeavyOverlap)
            {
                if (hasHeavyOverlap && !hasEmptyCluster && !hasVerySmallCluster && !hasVeryDominantCluster)
                {
                    Console.WriteLine($"  Note: Detected potentially overlapping distributions (centroid span: {overlapRatio:P0} of data range).");
                    Console.WriteLine("  K-means initialization may be less reliable. HMM will use temporal structure to refine.");
                }
                else
                {
                    Console.WriteLine("  Warning: K-means produced unbalanced clusters. Using robust initialization...");
                    kmeansAssignments = RobustInitialization(emissions, K, kmeans.Centroids);
                    
                    // Show the robust initialization stats
                    var robustCounts = new int[K];
                    foreach (int a in kmeansAssignments)
                        robustCounts[a]++;
                    
                    Console.WriteLine("  Robust initialization distribution:");
                    for (int k = 0; k < K; k++)
                    {
                        Console.WriteLine($"    State {k}: {robustCounts[k]} points ({100.0 * robustCounts[k] / T:F1}%)");
                    }
                }
            }
            
            model.InitialiseStatesFromAssignments(kmeansAssignments);
            
            Console.WriteLine("Running inference...");
            model.InferPosteriors();
            
            // Get inferred states using Viterbi algorithm for globally optimal sequence
            Console.WriteLine("Computing globally optimal state sequence using Viterbi algorithm...");
            int[] mapStates = model.GetViterbiStates();
            
            // Optionally compare with local (marginal) decoding
            if (false) // Set to true to see comparison
            {
                int[] localStates = model.GetMAPStates();
                int differences = 0;
                for (int t = 0; t < emissions.Length; t++)
                {
                    if (mapStates[t] != localStates[t])
                        differences++;
                }
                double diffPercent = (differences / (double)emissions.Length) * 100;
                Console.WriteLine($"  Viterbi vs Local decoding: {differences} differences ({diffPercent:F1}% of time steps)");
                Console.WriteLine();
            }

            // Print results
            Console.WriteLine();
            Console.WriteLine("=== RESULTS ===");
            Console.WriteLine();

            // State statistics
            var stateCounts = new int[K];
            for (int i = 0; i < mapStates.Length; i++)
            {
                stateCounts[mapStates[i]]++;
            }

            Console.WriteLine("State Distribution:");
            for (int k = 0; k < K; k++)
            {
                double percentage = (stateCounts[k] / (double)T) * 100;
                Console.WriteLine($"  State {k}: {stateCounts[k]} occurrences ({percentage:F1}%)");
            }
            Console.WriteLine();

            // State transitions - calculate counts and probabilities
            Console.WriteLine("State Transition Matrix:");
            Console.WriteLine("(rows: from state, columns: to state, values: probability)");
            Console.WriteLine();

            var transitionCounts = new int[K, K];
            for (int i = 1; i < mapStates.Length; i++)
            {
                transitionCounts[mapStates[i - 1], mapStates[i]]++;
            }

            // Calculate row totals for normalization
            int[] rowTotals = new int[K];
            for (int from = 0; from < K; from++)
            {
                for (int to = 0; to < K; to++)
                {
                    rowTotals[from] += transitionCounts[from, to];
                }
            }

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
                    if (rowTotals[from] > 0)
                    {
                        double prob = (double)transitionCounts[from, to] / rowTotals[from];
                        Console.Write($"{prob,8:F4}  ");
                    }
                    else
                    {
                        Console.Write($"{"N/A",8}  ");
                    }
                }
                Console.WriteLine($"  ({rowTotals[from]} transitions)");
            }
            Console.WriteLine();

            // Save results to output file
            string outputFile = Path.ChangeExtension(Path.GetFileName(inputFileName), ".results.csv");
            SaveResults(outputFile, emissions, mapStates);
            Console.WriteLine($"Results saved to '{outputFile}'");
        }

        /// <summary>
        /// Saves the results to a CSV file.
        /// </summary>
        /// <param name="filePath">Output file path.</param>
        /// <param name="emissions">Observation data.</param>
        /// <param name="states">Inferred states.</param>
        static void SaveResults(string filePath, double[] emissions, int[] states)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("# HMM Inference Results");
                writer.WriteLine("# Observation,InferredState");
                
                for (int i = 0; i < emissions.Length; i++)
                {
                    writer.WriteLine($"{emissions[i]},{states[i]}");
                }
            }
        }

        /// <summary>
        /// Runs the generation mode to create synthetic HMM sequences.
        /// </summary>
        /// <param name="args">Command-line arguments.</param>
        static void RunGenerateMode(string[] args)
        {
            // Parse arguments: --generate <output_file> <length> <num_states> [state_params...]
            if (args.Length < 4)
            {
                Console.WriteLine("Error: Insufficient arguments for generation mode.");
                Console.WriteLine();
                ShowUsage();
                return;
            }

            string outputFile = args[1];
            
            if (!int.TryParse(args[2], out int length) || length < 1)
            {
                Console.WriteLine("Error: Length must be a positive integer.");
                return;
            }

            if (!int.TryParse(args[3], out int numStates) || numStates < 2)
            {
                Console.WriteLine("Error: Number of states must be an integer >= 2.");
                return;
            }

            // Parse state parameters if provided
            double[] means = new double[numStates];
            double[] variances = new double[numStates];
            Random rand = new Random();

            if (args.Length >= 4 + numStates)
            {
                // User provided state parameters
                try
                {
                    for (int k = 0; k < numStates; k++)
                    {
                        string[] parts = args[4 + k].Split(',');
                        if (parts.Length != 2)
                        {
                            Console.WriteLine($"Error: State parameter {k} must be in format 'mean,variance'");
                            return;
                        }

                        if (!double.TryParse(parts[0], out means[k]))
                        {
                            Console.WriteLine($"Error: Invalid mean value for state {k}: '{parts[0]}'");
                            return;
                        }

                        if (!double.TryParse(parts[1], out variances[k]) || variances[k] <= 0)
                        {
                            Console.WriteLine($"Error: Invalid variance value for state {k}: '{parts[1]}' (must be > 0)");
                            return;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error parsing state parameters: {ex.Message}");
                    return;
                }
            }
            else
            {
                // Generate random parameters
                Console.WriteLine("No state parameters provided. Generating random parameters...");
                
                // Generate well-separated random means
                int baseMean = rand.Next(-50, 51);
                double separation = 10.0 + rand.NextDouble() * 10.0; // 10-20 units apart
                
                for (int k = 0; k < numStates; k++)
                {
                    means[k] = baseMean + k * separation;
                    variances[k] = 2.0 + rand.NextDouble() * 3.0; // Variance between 2 and 5
                }
            }

            // Display generation parameters
            Console.WriteLine("Generating synthetic HMM sequence...");
            Console.WriteLine($"  Output file: {outputFile}");
            Console.WriteLine($"  Length: {length}");
            Console.WriteLine($"  Number of states: {numStates}");
            Console.WriteLine();
            Console.WriteLine("State parameters:");
            for (int k = 0; k < numStates; k++)
            {
                Console.WriteLine($"  State {k}: mean = {means[k]:F2}, variance = {variances[k]:F2}, std dev = {Math.Sqrt(variances[k]):F2}");
            }
            Console.WriteLine();

            // Generate the sequence
            try
            {
                var (observations, states, transitionMatrix, initialProbs) = GenerateSyntheticSequence(length, numStates, means, variances, rand);
                
                // Save to file
                SaveGeneratedSequence(outputFile, observations, states, means, variances, transitionMatrix, initialProbs);
                
                // Display statistics
                Console.WriteLine("Generation complete!");
                Console.WriteLine();
                Console.WriteLine("Sequence statistics:");
                var stateCounts = new int[numStates];
                for (int i = 0; i < states.Length; i++)
                {
                    stateCounts[states[i]]++;
                }
                
                for (int k = 0; k < numStates; k++)
                {
                    double percentage = (stateCounts[k] / (double)length) * 100;
                    Console.WriteLine($"  State {k}: {stateCounts[k]} occurrences ({percentage:F1}%)");
                }
                Console.WriteLine();
                Console.WriteLine($"Data saved to '{outputFile}'");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating sequence: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"  {ex.InnerException.Message}");
                }
            }
        }

        /// <summary>
        /// Generates a synthetic HMM sequence.
        /// </summary>
        /// <param name="length">Number of observations to generate.</param>
        /// <param name="numStates">Number of hidden states.</param>
        /// <param name="means">Mean values for each state's emission distribution.</param>
        /// <param name="variances">Variance values for each state's emission distribution.</param>
        /// <param name="rand">Random number generator.</param>
        /// <returns>Tuple of observations, hidden states, transition matrix, and initial probabilities.</returns>
        static (double[] observations, int[] states, double[,] transitionMatrix, double[] initialProbs) GenerateSyntheticSequence(
            int length, int numStates, double[] means, double[] variances, Random rand)
        {
            double[] observations = new double[length];
            int[] states = new int[length];

            // Generate random transition matrix (uniform Dirichlet prior)
            double[,] transitionMatrix = new double[numStates, numStates];
            for (int from = 0; from < numStates; from++)
            {
                double sum = 0;
                for (int to = 0; to < numStates; to++)
                {
                    // Sample from uniform distribution
                    transitionMatrix[from, to] = rand.NextDouble();
                    sum += transitionMatrix[from, to];
                }
                // Normalize
                for (int to = 0; to < numStates; to++)
                {
                    transitionMatrix[from, to] /= sum;
                }
            }

            // Generate random initial state distribution
            double[] initialProbs = new double[numStates];
            double initialSum = 0;
            for (int k = 0; k < numStates; k++)
            {
                initialProbs[k] = rand.NextDouble();
                initialSum += initialProbs[k];
            }
            for (int k = 0; k < numStates; k++)
            {
                initialProbs[k] /= initialSum;
            }

            // Generate sequence
            states[0] = SampleFromDiscrete(initialProbs, rand);
            observations[0] = SampleFromGaussian(means[states[0]], Math.Sqrt(variances[states[0]]), rand);

            for (int t = 1; t < length; t++)
            {
                // Sample next state based on transition probabilities
                double[] transProbs = new double[numStates];
                for (int k = 0; k < numStates; k++)
                {
                    transProbs[k] = transitionMatrix[states[t - 1], k];
                }
                states[t] = SampleFromDiscrete(transProbs, rand);

                // Sample observation from emission distribution
                observations[t] = SampleFromGaussian(means[states[t]], Math.Sqrt(variances[states[t]]), rand);
            }

            return (observations, states, transitionMatrix, initialProbs);
        }

        /// <summary>
        /// Samples from a discrete distribution.
        /// </summary>
        /// <param name="probabilities">Probability vector (must sum to 1).</param>
        /// <param name="rand">Random number generator.</param>
        /// <returns>Sampled index.</returns>
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

        /// <summary>
        /// Samples from a Gaussian distribution using Box-Muller transform.
        /// </summary>
        /// <param name="mean">Mean of the distribution.</param>
        /// <param name="stdDev">Standard deviation of the distribution.</param>
        /// <param name="rand">Random number generator.</param>
        /// <returns>Sampled value.</returns>
        static double SampleFromGaussian(double mean, double stdDev, Random rand)
        {
            // Box-Muller transform
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return mean + stdDev * z0;
        }

        /// <summary>
        /// Saves the generated sequence to a CSV file.
        /// </summary>
        /// <param name="filePath">Output file path.</param>
        /// <param name="observations">Generated observations.</param>
        /// <param name="states">True hidden states.</param>
        /// <param name="means">Mean values for each state.</param>
        /// <param name="variances">Variance values for each state.</param>
        /// <param name="transitionMatrix">True transition matrix.</param>
        /// <param name="initialProbs">True initial state probabilities.</param>
        static void SaveGeneratedSequence(string filePath, double[] observations, int[] states, 
            double[] means, double[] variances, double[,] transitionMatrix, double[] initialProbs)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("# Synthetic HMM Sequence");
                writer.WriteLine($"# Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                writer.WriteLine($"# Length: {observations.Length}");
                writer.WriteLine($"# Number of states: {means.Length}");
                writer.WriteLine("#");
                writer.WriteLine("# State parameters:");
                for (int k = 0; k < means.Length; k++)
                {
                    writer.WriteLine($"#   State {k}: mean={means[k]:F2}, variance={variances[k]:F2}");
                }
                writer.WriteLine("#");
                writer.WriteLine("# Initial state probabilities:");
                for (int k = 0; k < initialProbs.Length; k++)
                {
                    writer.WriteLine($"#   P(State {k}) = {initialProbs[k]:F4}");
                }
                writer.WriteLine("#");
                writer.WriteLine("# Transition matrix:");
                writer.WriteLine("# (rows: from state, columns: to state, values: probability)");
                writer.WriteLine("#");
                
                // Print header
                writer.Write("#         ");
                for (int to = 0; to < means.Length; to++)
                {
                    writer.Write($"State {to}  ");
                }
                writer.WriteLine();
                
                writer.Write("#         ");
                for (int to = 0; to < means.Length; to++)
                {
                    writer.Write("--------  ");
                }
                writer.WriteLine();
                
                // Print matrix rows
                for (int from = 0; from < means.Length; from++)
                {
                    writer.Write($"# State {from} ");
                    for (int to = 0; to < means.Length; to++)
                    {
                        writer.Write($"{transitionMatrix[from, to],8:F4}  ");
                    }
                    writer.WriteLine();
                }
                writer.WriteLine("#");
                writer.WriteLine("# Format: observation");
                writer.WriteLine("#");
                
                for (int i = 0; i < observations.Length; i++)
                {
                    writer.WriteLine($"{observations[i]:F4}");
                }
            }

            // Also save the true states to a separate file for validation
            string statesFile = Path.ChangeExtension(filePath, ".states.csv");
            using (StreamWriter writer = new StreamWriter(statesFile))
            {
                writer.WriteLine("# True hidden states for generated sequence");
                writer.WriteLine("# Index,TrueState");
                for (int i = 0; i < states.Length; i++)
                {
                    writer.WriteLine($"{i},{states[i]}");
                }
            }
            
            Console.WriteLine($"True states saved to '{statesFile}' for validation");
        }
    }
}
