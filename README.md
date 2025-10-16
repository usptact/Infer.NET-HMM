# Hidden Markov Models - Infer.NET Implementation

Two powerful Bayesian inference tools for sequential data analysis using the [Infer.NET framework](https://dotnet.github.io/infer/):

1. **Hidden Markov Model (HMM)** - Single latent chain for standard time series
2. **Factorial Hidden Markov Model (FHMM)** - Multiple independent latent chains for complex decomposition

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Hidden Markov Model (Single Chain)](#hidden-markov-model-single-chain)
- [Factorial Hidden Markov Model (Multiple Chains)](#factorial-hidden-markov-model-multiple-chains)
- [When to Use Which Model](#when-to-use-which-model)
- [Building](#building)

---

## Overview

### What is a Hidden Markov Model?

A [Hidden Markov Model](http://en.wikipedia.org/wiki/Hidden_markov_model) models sequential data where:
- **Hidden states** follow a Markov chain (transitions depend only on previous state)
- Each state **emits observations** from a Gaussian distribution
- Goal: Infer hidden states and parameters from observed data

### What is a Factorial Hidden Markov Model?

A Factorial HMM extends the standard HMM with **multiple independent chains**:
- Each chain evolves with its own states and transitions
- **Observations are the SUM** of emissions from all chains plus noise
- Useful for decomposing complex signals into multiple independent sources

**Example Use Cases:**
- **Financial data**: Trend + Seasonal + Noise components
- **Signal processing**: Multiple independent signal sources
- **System monitoring**: Multiple subsystems contributing to aggregate metric

<img src="https://raw.githubusercontent.com/oliparson/infer-hmm/master/bayes-hmm.png" alt="Bayesian HMM graphical model" style="width: 25% height: 25%;"/>

---

## Requirements

- **.NET 8.0** SDK
- **Microsoft.ML.Probabilistic 0.4.2504.701** (auto-installed via NuGet)

```bash
# Verify .NET installation
dotnet --version

# Clone and build
git clone <repository-url>
cd infernet-hmm
dotnet build
```

---

## Hidden Markov Model (Single Chain)

For standard time series with one sequence of hidden states.

### Quick Start

```bash
# Show usage
dotnet run --project HiddenMarkovModel

# Run inference on sample data
dotnet run --project HiddenMarkovModel sample_data.csv 2

# Generate synthetic test data
dotnet run --project HiddenMarkovModel -- -g synthetic.csv 100 2 10,3 20,4

# Run inference on generated data
dotnet run --project HiddenMarkovModel synthetic.csv 2
```

### Inference Mode

Analyze existing time series data to discover hidden states.

**Usage:**
```bash
dotnet run --project HiddenMarkovModel <csv_file> [num_states]
```

**Arguments:**
- `csv_file` - Path to CSV file with observations (one value per line)
- `num_states` - Number of hidden states (default: 2)

**Practical Examples:**
```bash
# Binary state detection (e.g., machine on/off, normal/anomaly)
dotnet run --project HiddenMarkovModel sensor_data.csv 2

# Market regime detection (bullish/bearish/sideways)
dotnet run --project HiddenMarkovModel stock_returns.csv 3

# System with multiple operating modes
dotnet run --project HiddenMarkovModel system_metrics.csv 4
```

**Input Format:**
```csv
# Temperature sensor readings
23.5
24.1
38.7
39.2
23.8
...
```
- One numeric value per line
- Lines starting with `#` are comments (ignored)
- Empty lines are ignored

**Output:**

Console shows:
```
=== RESULTS ===

State Distribution:
  State 0: 487 occurrences (48.7%)
  State 1: 513 occurrences (51.3%)

State Transition Matrix:
(rows: from state, columns: to state, values: probability)

        State 0  State 1  
        --------  --------  
State 0   0.4630    0.5370    (486 transitions)
State 1   0.5088    0.4912    (513 transitions)
```

File `<input>.results.csv`:
```csv
# HMM Inference Results
# Observation,InferredState
23.5,0
24.1,0
38.7,1
39.2,1
...
```

**Key Features:**
- ✅ Automatic K-means initialization for robust state discovery
- ✅ Viterbi algorithm for globally optimal state sequences
- ✅ Handles overlapping and imbalanced states
- ✅ Data-driven priors with symmetry breaking

**Interpretation:**
- **State Distribution**: Frequency of each state
- **Transition Matrix**: Probability of switching between states
  - High diagonal = stable states (low switching)
  - Off-diagonal = transition probabilities

### Generation Mode

Create synthetic HMM data with known parameters for testing and validation.

**Usage:**
```bash
dotnet run --project HiddenMarkovModel -- -g <output> <length> <states> [params...]
```

**Arguments:**
- `output` - Output CSV filename
- `length` - Number of observations to generate
- `states` - Number of hidden states (≥ 2)
- `params` - Optional: `mean,variance` for each state (random if omitted)

**Examples:**
```bash
# Random well-separated parameters
dotnet run --project HiddenMarkovModel -- -g random.csv 500 2

# Specified parameters (well-separated)
dotnet run --project HiddenMarkovModel -- -g easy.csv 200 2 10,2 30,2

# Overlapping states (challenging)
dotnet run --project HiddenMarkovModel -- -g hard.csv 200 2 10,8 15,8

# Multi-state system
dotnet run --project HiddenMarkovModel -- -g multi.csv 1000 4 5,1 15,2 25,3 35,1
```

**Output Files:**

1. **`<output>.csv`** - Observations with metadata header containing true parameters
2. **`<output>.states.csv`** - True hidden state sequence for validation

**Parameter Guidelines:**
- **Mean separation**: `|mean₁ - mean₂| / √variance` > 3 for easy separation
- **Variance**: 
  - Low (1-5): Tight clusters, easy inference
  - Medium (5-15): Realistic overlap
  - High (15+): Heavy overlap, challenging

**Validation Workflow:**
```bash
# 1. Generate with known parameters
dotnet run --project HiddenMarkovModel -- -g test.csv 500 2 10,3 25,3

# 2. Run inference
dotnet run --project HiddenMarkovModel test.csv 2

# 3. Compare results:
#    - Check inferred means vs true (10, 25)
#    - Compare test.results.csv with test.states.csv
#    - Evaluate state recovery accuracy
```

---

## Factorial Hidden Markov Model (Multiple Chains)

For complex time series requiring decomposition into multiple independent latent processes.

### Quick Start

```bash
# Show usage
dotnet run --project FactorialHiddenMarkovModel

# Run inference: 2 chains with 2 states each
dotnet run --project FactorialHiddenMarkovModel data.csv 2 2 2

# Generate synthetic factorial data
dotnet run --project FactorialHiddenMarkovModel -- -g fhmm.csv 200 2 2 2 -5,2 5,2 -3,1 3,1

# Multiple runs for robust inference (RECOMMENDED)
bash run_multiple.sh data.csv 2 2 2 10
```

### Inference Mode

Decompose observations into multiple independent latent chains.

**Usage:**
```bash
dotnet run --project FactorialHiddenMarkovModel <csv_file> <num_chains> <states_per_chain...>
```

**Arguments:**
- `csv_file` - Path to CSV file with observations
- `num_chains` - Number of independent chains (2-3 supported)
- `states_per_chain` - Number of states for each chain (space-separated)

**Practical Examples:**
```bash
# Financial decomposition: trend (2 states) + seasonal (2 states)
dotnet run --project FactorialHiddenMarkovModel prices.csv 2 2 2

# Signal with 3 sources: 2, 3, and 2 states
dotnet run --project FactorialHiddenMarkovModel signal.csv 3 2 3 2

# Asymmetric chains
dotnet run --project FactorialHiddenMarkovModel data.csv 2 3 2
```

**Input Format:**
Same as standard HMM - one observation per line.

**Output:**

Console shows:
```
=== RESULTS ===

Model Evidence (log): -33102.90

Inferred emission means by chain:
Chain 0:
  State 0: mean=5.42
  State 1: mean=14.49
Chain 1:
  State 0: mean=-0.87
  State 1: mean=-13.36

Observation noise precision: 0.153 (variance: 6.54)

Chain 0 State Distribution:
  State 0: 7647 occurrences (76.5%)
  State 1: 2353 occurrences (23.5%)

Chain 1 State Distribution:
  State 0: 4430 occurrences (44.3%)
  State 1: 5570 occurrences (55.7%)
```

Files created:
- **`<input>.factorial.csv`** - Observations with all inferred chain states
- **`<input>.chains.csv`** - Individual chain contributions

**Key Features:**
- ✅ **Viterbi algorithm** for globally optimal state paths (not local marginals!)
- ✅ **Randomized priors** for better exploration across runs
- ✅ **Model evidence** for solution quality assessment
- ✅ Handles 2-3 independent chains

**⚠️ Important: Multiple Runs Required**

Factorial HMM inference has **inherent challenges**:
1. **Identifiability**: Multiple state combinations produce similar observations
2. **Local optima**: Different initializations → different solutions
3. **Label switching**: Solutions equivalent up to state permutations

**Solution: Run 10-20 times, select best by model evidence**

### Multiple Runs Script

Use the provided `run_multiple.sh` script:

```bash
# Run 10 times, automatically select best
bash run_multiple.sh factorial_data.csv 2 2 2 10

# Output:
# - Creates multi_run_<timestamp>/ directory
# - Saves all run outputs
# - Identifies best run by model evidence
# - Copies best results to factorial_data.factorial_best.csv
```

**Manual approach:**
```bash
# Run multiple times
for i in {1..10}; do
    dotnet run --project FactorialHiddenMarkovModel data.csv 2 2 2 > run_$i.txt
done

# Compare model evidence
grep "Model Evidence" run_*.txt

# Example output:
# run_1.txt:Model Evidence (log): -34367.99
# run_2.txt:Model Evidence (log): -33102.64  ← Better
# run_5.txt:Model Evidence (log): -33101.82  ← Best!

# Use the one with HIGHEST (least negative) evidence
```

**When to trust results:**
- ✅ Model evidence consistent across runs (within ±20-30)
- ✅ Inferred means well-separated (|mean_i - mean_j| > 3)
- ✅ State assignments reasonable (not all one state)
- ✅ Observation precision reasonable

**Red flags:**
- ⚠️ Model evidence varies wildly (±100+)
- ⚠️ Means collapse to similar values
- ⚠️ Observation variance >> data variance

### Generation Mode

Create synthetic factorial HMM data for testing.

**Usage:**
```bash
dotnet run --project FactorialHiddenMarkovModel -- -g <output> <length> <chains> <states...> [params...]
```

**Arguments:**
- `output` - Output CSV filename
- `length` - Number of observations
- `chains` - Number of chains (2-3)
- `states` - States per chain (space-separated)
- `params` - Optional: `mean,variance` for each state in each chain

**Examples:**
```bash
# Random parameters: 2 chains × 2 states
dotnet run --project FactorialHiddenMarkovModel -- -g fhmm.csv 200 2 2 2

# Specified parameters: 2 chains × 2 states
dotnet run --project FactorialHiddenMarkovModel -- -g fhmm.csv 200 2 2 2 -5,2 5,2 -3,1 3,1
# Interpretation: Chain0: State0(-5,var=2), State1(5,var=2)
#                 Chain1: State0(-3,var=1), State1(3,var=1)

# 3 chains
dotnet run --project FactorialHiddenMarkovModel -- -g fhmm3.csv 300 3 2 2 2 -10,3 10,3 -5,2 5,2 -3,1 3,1
```

**Output Files:**

1. **`<output>.csv`** - Observations with full metadata
2. **`<output>.chains.csv`** - True state sequences for all chains

**Validation Workflow:**
```bash
# 1. Generate with known parameters
dotnet run --project FactorialHiddenMarkovModel -- -g test.csv 200 2 2 2 -5,2 5,2 -3,1 3,1

# 2. Run inference multiple times
bash run_multiple.sh test.csv 2 2 2 10

# 3. Compare best result:
#    - test.chains.csv (true states)
#    - test.factorial_best.csv (inferred states)
#    - Check if means close to true: Chain0=[-5,5], Chain1=[-3,3]
```

### Understanding Model Evidence

**Model Evidence (log marginal likelihood)** quantifies how well the model explains the data:

- **Higher (less negative)** = better fit
- Balances fit quality AND model complexity
- Essential for comparing runs

**Typical differences:**
- **±10-20**: Marginal improvement
- **±50+**: Significant improvement
- **±200+**: One run clearly failed

**Example:**
```
Run 1: -34367.99  (bad local optimum)
Run 2: -33102.64  (good)
Run 5: -33101.82  (best!)
```
→ Run 5 is ~1266 units better than Run 1 (use Run 5!)

### Advanced: Comparing Viterbi vs Local Decoding

To see the difference between global (Viterbi) and local (marginal) decoding:

1. In `FactorialHiddenMarkovModel/Program.cs`, change line 283:
   ```csharp
   if (true) // Set to true to see comparison
   ```

2. Run inference:
   ```bash
   dotnet run --project FactorialHiddenMarkovModel data.csv 2 2 2
   ```

3. Output will show:
   ```
   Comparing Viterbi (global) vs Marginal (local) decoding:
     Chain 0: 730 differences (7.3% of time steps)
     Chain 1: 309 differences (3.1% of time steps)
   ```

**Interpretation:** Viterbi changes 3-7% of state assignments by considering temporal dependencies!

---

## When to Use Which Model

### Use Hidden Markov Model (Single Chain) when:

- ✅ Data has **one dominant latent process**
- ✅ Looking for **regime changes** or **state switches**
- ✅ Need **fast, reliable inference**
- ✅ Examples:
  - Machine on/off states
  - Market regimes (bull/bear)
  - Anomaly detection (normal/abnormal)
  - Speech recognition phonemes

**Characteristics:**
- Fast inference (50 iterations, ~1-5 seconds)
- Robust with K-means initialization
- Reliable for 2-5 states
- Works with 50+ observations

### Use Factorial Hidden Markov Model (Multiple Chains) when:

- ✅ Data is **sum of multiple independent sources**
- ✅ Need to **decompose** signal into components
- ✅ Can run **multiple inference attempts**
- ✅ Examples:
  - Trend + Seasonal + Noise decomposition
  - Multiple sensor sources
  - Multi-factor financial models
  - Additive signal sources

**Characteristics:**
- Slower inference (75 iterations, ~30-60 seconds)
- Requires multiple runs (10-20 recommended)
- Challenging with overlapping components
- Needs 200+ observations for reliability
- Model evidence selection essential

### Decision Tree

```
Is your data the SUM of multiple independent processes?
│
├─ NO → Use Hidden Markov Model
│        (Standard HMM for single latent chain)
│
└─ YES → Use Factorial Hidden Markov Model
         (Multiple independent chains)
         ⚠️  Remember: Run 10-20 times, pick best evidence!
```

---

## Building

### Development Build
```bash
# Build both projects
dotnet build

# Build specific project
dotnet build HiddenMarkovModel/HiddenMarkovModel.csproj
dotnet build FactorialHiddenMarkovModel/FactorialHiddenMarkovModel.csproj
```

### Production Build
```bash
# Create standalone executables (no .NET runtime required)

# For macOS (Apple Silicon)
dotnet publish HiddenMarkovModel -c Release -r osx-arm64 --self-contained
dotnet publish FactorialHiddenMarkovModel -c Release -r osx-arm64 --self-contained

# For macOS (Intel)
dotnet publish HiddenMarkovModel -c Release -r osx-x64 --self-contained

# For Linux
dotnet publish HiddenMarkovModel -c Release -r linux-x64 --self-contained

# For Windows
dotnet publish HiddenMarkovModel -c Release -r win-x64 --self-contained
```

Executables will be in `bin/Release/net8.0/<runtime>/publish/`

---

## Tips and Best Practices

### For Both Models

1. **Data preparation**:
   - Remove obvious outliers
   - Consider normalization for numerical stability
   - Check for NaN/Inf values

2. **Number of states**:
   - Start with 2-3 states
   - More states need more data
   - Use domain knowledge when possible

3. **Data size**:
   - HMM: 50-100+ observations
   - FHMM: 200+ observations

### For Factorial HMM Specifically

1. **Always run multiple times** (10-20 runs)
2. **Use model evidence** to select best run
3. **Start with well-separated test data** to verify inference works
4. **Check means are well-separated** in results (|mean_i - mean_j| > 3)
5. **Validate on synthetic data** before real data
6. **Use 2 chains initially**, add 3rd chain only if necessary
7. **Expect challenges** - FHMM inference is inherently difficult!

---

## Acknowledgments

- [Microsoft Research](https://www.microsoft.com/en-us/research/) for the Infer.NET framework
- [Matteo Venanzi](http://users.ecs.soton.ac.uk/mv1g10/) for expertise in chain model efficiency
- Original HMM implementation inspiration from the Infer.NET community

---

## License

This project uses the MIT License. See LICENSE file for details.

Infer.NET is licensed under the MIT License by Microsoft Research.
