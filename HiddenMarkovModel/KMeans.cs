using System;
using System.Linq;

namespace HiddenMarkovModel
{
    /// <summary>
    /// K-means clustering for 1D data.
    /// </summary>
    public class KMeans
    {
        private readonly int numClusters;
        private readonly int maxIterations;
        private readonly Random random;

        /// <summary>
        /// Gets the final cluster centroids after fitting.
        /// </summary>
        public double[] Centroids { get; private set; }

        /// <summary>
        /// Gets the cluster assignments for each data point.
        /// </summary>
        public int[] Assignments { get; private set; }

        /// <summary>
        /// Gets the number of iterations used for convergence.
        /// </summary>
        public int IterationsUsed { get; private set; }

        /// <summary>
        /// Initializes a new instance of the KMeans class.
        /// </summary>
        /// <param name="numClusters">Number of clusters.</param>
        /// <param name="maxIterations">Maximum number of iterations (default: 100).</param>
        /// <param name="randomSeed">Random seed for reproducibility (default: 42).</param>
        public KMeans(int numClusters, int maxIterations = 100, int randomSeed = 42)
        {
            if (numClusters < 1)
                throw new ArgumentException("Number of clusters must be at least 1", nameof(numClusters));
            if (maxIterations < 1)
                throw new ArgumentException("Max iterations must be at least 1", nameof(maxIterations));

            this.numClusters = numClusters;
            this.maxIterations = maxIterations;
            this.random = new Random(randomSeed);
            
            Centroids = new double[numClusters];
            Assignments = Array.Empty<int>();
            IterationsUsed = 0;
        }

        /// <summary>
        /// Fits the K-means model to the data.
        /// </summary>
        /// <param name="data">The 1D data to cluster.</param>
        /// <param name="verbose">Whether to print progress information.</param>
        /// <returns>The cluster assignments for each data point.</returns>
        public int[] Fit(double[] data, bool verbose = false)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Data cannot be null or empty", nameof(data));
            if (data.Length < numClusters)
                throw new ArgumentException($"Data length ({data.Length}) must be at least the number of clusters ({numClusters})", nameof(data));

            int N = data.Length;
            Assignments = new int[N];
            
            // Initialize centroids using quantiles
            InitializeCentroidsFromQuantiles(data);
            
            if (verbose)
            {
                Console.WriteLine("  Initial K-means centroids:");
                for (int k = 0; k < numClusters; k++)
                {
                    Console.WriteLine($"    Cluster {k}: {Centroids[k]:F4}");
                }
            }
            
            // K-means iterations
            bool changed = true;
            IterationsUsed = 0;
            
            while (changed && IterationsUsed < maxIterations)
            {
                changed = false;
                IterationsUsed++;
                
                // Assignment step: assign each point to nearest centroid
                for (int i = 0; i < N; i++)
                {
                    int bestCluster = FindNearestCentroid(data[i]);
                    
                    if (Assignments[i] != bestCluster)
                    {
                        Assignments[i] = bestCluster;
                        changed = true;
                    }
                }
                
                // Update step: recompute centroids
                UpdateCentroids(data);
            }
            
            if (verbose)
            {
                Console.WriteLine($"  K-means converged in {IterationsUsed} iterations");
                Console.WriteLine("  Final centroids:");
                for (int k = 0; k < numClusters; k++)
                {
                    int count = Assignments.Count(a => a == k);
                    Console.WriteLine($"    Cluster {k}: {Centroids[k]:F4} ({count} points)");
                }
            }
            
            return Assignments;
        }

        /// <summary>
        /// Predicts cluster assignments for new data points.
        /// </summary>
        /// <param name="data">The data points to assign to clusters.</param>
        /// <returns>Cluster assignments.</returns>
        public int[] Predict(double[] data)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Data cannot be null or empty", nameof(data));
            if (Centroids == null || Centroids.Length == 0)
                throw new InvalidOperationException("Model must be fitted before prediction");

            int[] predictions = new int[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                predictions[i] = FindNearestCentroid(data[i]);
            }
            
            return predictions;
        }

        /// <summary>
        /// Initializes centroids using quantiles from the sorted data.
        /// This provides a good spread across the data range.
        /// </summary>
        /// <param name="data">The data to initialize from.</param>
        private void InitializeCentroidsFromQuantiles(double[] data)
        {
            double[] sortedData = data.OrderBy(x => x).ToArray();
            
            for (int k = 0; k < numClusters; k++)
            {
                // Place centroids at quantile positions
                int quantileIndex = (int)((k + 0.5) * sortedData.Length / numClusters);
                if (quantileIndex >= sortedData.Length)
                    quantileIndex = sortedData.Length - 1;
                
                Centroids[k] = sortedData[quantileIndex];
            }
        }

        /// <summary>
        /// Finds the nearest centroid for a given data point.
        /// </summary>
        /// <param name="value">The data point.</param>
        /// <returns>Index of the nearest centroid.</returns>
        private int FindNearestCentroid(double value)
        {
            int bestCluster = 0;
            double minDistance = Math.Abs(value - Centroids[0]);
            
            for (int k = 1; k < numClusters; k++)
            {
                double distance = Math.Abs(value - Centroids[k]);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    bestCluster = k;
                }
            }
            
            return bestCluster;
        }

        /// <summary>
        /// Updates centroids based on current assignments.
        /// </summary>
        /// <param name="data">The data points.</param>
        private void UpdateCentroids(double[] data)
        {
            int[] clusterCounts = new int[numClusters];
            double[] clusterSums = new double[numClusters];
            
            for (int i = 0; i < data.Length; i++)
            {
                int cluster = Assignments[i];
                clusterSums[cluster] += data[i];
                clusterCounts[cluster]++;
            }
            
            for (int k = 0; k < numClusters; k++)
            {
                if (clusterCounts[k] > 0)
                {
                    Centroids[k] = clusterSums[k] / clusterCounts[k];
                }
                // If a cluster has no points, keep the old centroid
            }
        }

        /// <summary>
        /// Gets the cluster counts for each cluster.
        /// </summary>
        /// <returns>Array of counts for each cluster.</returns>
        public int[] GetClusterCounts()
        {
            if (Assignments == null || Assignments.Length == 0)
                throw new InvalidOperationException("Model must be fitted before getting cluster counts");

            int[] counts = new int[numClusters];
            foreach (int assignment in Assignments)
            {
                counts[assignment]++;
            }
            
            return counts;
        }
    }
}

