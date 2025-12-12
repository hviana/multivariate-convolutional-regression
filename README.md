Model: # 🧠 ConvolutionalRegression

<div align="center">

**High-Performance Convolutional Neural Network for Multivariate Regression with
Incremental Online Learning**

[Features](#-features) • [Quick Start](#-quick-start) •
[Architecture](#-architecture) • [API Reference](#-api-reference) •
[Parameters](#-configuration-parameters)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [📖 API Reference](#-api-reference)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
- [🔧 Parameter Optimization Guide](#-parameter-optimization-guide)
- [📊 Use Case Examples](#-use-case-examples)
- [🧮 Mathematical Foundations](#-mathematical-foundations)
- [🎯 Best Practices](#-best-practices)
- [⚠️ Troubleshooting](#️-troubleshooting)
- [📈 Performance Tips](#-performance-tips)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔷 Core Neural Network

- **Conv1D Layers** with same padding
- **ReLU Activation** for non-linearity
- **Dense Output Layer** for regression
- **He Initialization** for optimal weight starting

</td>
<td width="50%">

### ⚡ Online Learning

- **Incremental Training** - learn sample by sample
- **Adam Optimizer** with momentum
- **Cosine Warmup** learning rate schedule
- **Adaptive Learning** without full retraining

</td>
</tr>
<tr>
<td width="50%">

### 📊 Normalization & Statistics

- **Welford's Algorithm** for running statistics
- **Z-Score Normalization** computed online
- **No Data Storage** required for normalization
- **Numerically Stable** computations

</td>
<td width="50%">

### 🛡️ Robustness Features

- **L2 Regularization** prevents overfitting
- **Outlier Detection** & downweighting
- **ADWIN Drift Detection** for concept drift
- **Uncertainty Quantification** with confidence intervals

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Basic Usage

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// 1️⃣ Create model with default configuration
const model = new ConvolutionalRegression();

// 2️⃣ Prepare training data
const trainingData = {
  xCoordinates: [
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
  ],
  yCoordinates: [
    [4.0],
    [5.0],
    [6.0],
    [7.0],
  ],
};

// 3️⃣ Train incrementally
const result = model.fitOnline(trainingData);
console.log(`📉 Loss: ${result.loss.toFixed(6)}`);
console.log(`📈 Learning Rate: ${result.effectiveLearningRate.toFixed(6)}`);

// 4️⃣ Generate predictions
const predictions = model.predict(5);
predictions.predictions.forEach((pred, i) => {
  console.log(
    `Step ${i + 1}: ${pred.predicted[0].toFixed(4)} ± ${
      pred.standardError[0].toFixed(4)
    }`,
  );
});
```

### Output Example

```
📉 Loss: 0.023451
📈 Learning Rate: 0.000040
Step 1: 7.9823 ± 0.1234
Step 2: 8.9756 ± 0.1567
Step 3: 9.9634 ± 0.1823
Step 4: 10.9512 ± 0.2134
Step 5: 11.9389 ± 0.2456
```

---

## 🏗️ Architecture

### Network Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONVOLUTIONAL REGRESSION NETWORK                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────────────────────────────┐    ┌────────┐ │
│   │   INPUT     │    │     HIDDEN CONVOLUTIONAL LAYERS     │    │ OUTPUT │ │
│   │   LAYER     │    │                                     │    │ LAYER  │ │
│   └─────────────┘    └─────────────────────────────────────┘    └────────┘ │
│                                                                              │
│   ┌───────────┐      ┌─────────┐      ┌─────────┐              ┌─────────┐ │
│   │           │      │ Conv1D  │      │ Conv1D  │              │  Dense  │ │
│   │  Input    │─────▶│    +    │─────▶│    +    │─────▶ ... ──▶│  Layer  │ │
│   │ (inputDim)│      │  ReLU   │      │  ReLU   │              │         │ │
│   │           │      │         │      │         │              │         │ │
│   └───────────┘      └─────────┘      └─────────┘              └─────────┘ │
│        │                  │                │                        │       │
│        ▼                  ▼                ▼                        ▼       │
│   [1 × spatial]    [filters × spatial]  [filters × spatial]   [outputDim]  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Architecture Formula:
Input(inputDim) → [Conv1D(filters, kernelSize, same) → ReLU]×L → Flatten → Dense(outputDim)
```

### Data Flow Diagram

```
                        TRAINING PIPELINE
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  ┌──────────┐   ┌────────────┐   ┌─────────┐   ┌──────────────┐ │
│  │   Raw    │   │  Welford   │   │  Z-Score │   │  Normalized  │ │
│  │   Data   │──▶│  Update    │──▶│  Norm    │──▶│    Data      │ │
│  │ (x, y)   │   │ (μ, σ²)    │   │          │   │   (x̃, ỹ)     │ │
│  └──────────┘   └────────────┘   └─────────┘   └──────────────┘ │
│                                                        │         │
│                    ┌───────────────────────────────────┘         │
│                    ▼                                             │
│           ┌─────────────────┐                                    │
│           │  Forward Pass   │                                    │
│           │  Conv→ReLU→Dense│                                    │
│           └────────┬────────┘                                    │
│                    │                                             │
│                    ▼                                             │
│           ┌─────────────────┐   ┌─────────────┐                 │
│           │  Compute Loss   │──▶│   Outlier   │                 │
│           │  MSE + L2 Reg   │   │  Detection  │                 │
│           └────────┬────────┘   └─────────────┘                 │
│                    │                                             │
│                    ▼                                             │
│           ┌─────────────────┐   ┌─────────────┐                 │
│           │  Backward Pass  │──▶│    ADWIN    │                 │
│           │  Compute ∇L     │   │  Drift Det  │                 │
│           └────────┬────────┘   └─────────────┘                 │
│                    │                                             │
│                    ▼                                             │
│           ┌─────────────────┐                                    │
│           │   Adam Update   │                                    │
│           │  with Warmup    │                                    │
│           └─────────────────┘                                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📖 API Reference

### Constructor

```typescript
const model = new ConvolutionalRegression(config?: ConvolutionalRegressionConfig);
```

### Main Methods

| Method                    | Description                  | Returns              |
| ------------------------- | ---------------------------- | -------------------- |
| `fitOnline(data)`         | Incremental online training  | `FitResult`          |
| `predict(steps)`          | Generate future predictions  | `PredictionResult`   |
| `getModelSummary()`       | Get model state summary      | `ModelSummary`       |
| `getWeights()`            | Export model weights         | `WeightInfo`         |
| `getNormalizationStats()` | Get normalization statistics | `NormalizationStats` |
| `reset()`                 | Reset model to initial state | `void`               |

### Interfaces

<details>
<summary><b>📥 FitInput</b> - Training data structure</summary>

```typescript
interface FitInput {
  /** Input features: [numSamples][inputDim] */
  xCoordinates: number[][];
  /** Target outputs: [numSamples][outputDim] */
  yCoordinates: number[][];
}
```

**Example:**

```typescript
const data: FitInput = {
  xCoordinates: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  yCoordinates: [[10, 11], [12, 13], [14, 15]],
};
```

</details>

<details>
<summary><b>📤 FitResult</b> - Training step result</summary>

```typescript
interface FitResult {
  loss: number; // Current MSE loss value
  gradientNorm: number; // L2 norm of gradient vector
  effectiveLearningRate: number; // LR after warmup/decay
  isOutlier: boolean; // Sample flagged as outlier
  converged: boolean; // Model has converged
  sampleIndex: number; // Index of processed sample
  driftDetected: boolean; // Concept drift detected
}
```

</details>

<details>
<summary><b>🔮 PredictionResult</b> - Prediction output</summary>

```typescript
interface PredictionResult {
  predictions: SinglePrediction[]; // Predictions for each step
  accuracy: number; // Model accuracy: 1/(1 + avgLoss)
  sampleCount: number; // Total samples processed
  isModelReady: boolean; // Model ready for prediction
}

interface SinglePrediction {
  predicted: number[]; // Point estimate
  lowerBound: number[]; // Lower 95% CI
  upperBound: number[]; // Upper 95% CI
  standardError: number[]; // Standard error per dimension
}
```

</details>

<details>
<summary><b>📊 ModelSummary</b> - Model state overview</summary>

```typescript
interface ModelSummary {
  isInitialized: boolean; // Network initialized
  inputDimension: number; // Auto-detected input dim
  outputDimension: number; // Auto-detected output dim
  hiddenLayers: number; // Number of conv layers
  convolutionsPerLayer: number; // Filters per layer
  kernelSize: number; // Convolution kernel size
  totalParameters: number; // Total trainable params
  sampleCount: number; // Samples processed
  accuracy: number; // Current accuracy metric
  converged: boolean; // Training converged
  effectiveLearningRate: number; // Current learning rate
  driftCount: number; // Detected drift events
}
```

</details>

---

## ⚙️ Configuration Parameters

### Complete Parameter Reference

```typescript
interface ConvolutionalRegressionConfig {
  // Network Architecture
  hiddenLayers?: number; // 1-10, default: 2
  convolutionsPerLayer?: number; // 1-256, default: 32
  kernelSize?: number; // ≥1, default: 3

  // Adam Optimizer
  learningRate?: number; // >0, default: 0.001
  warmupSteps?: number; // ≥0, default: 100
  totalSteps?: number; // ≥1, default: 10000
  beta1?: number; // 0-0.9999, default: 0.9
  beta2?: number; // 0-0.9999, default: 0.999
  epsilon?: number; // >0, default: 1e-8

  // Regularization
  regularizationStrength?: number; // ≥0, default: 1e-4
  convergenceThreshold?: number; // ≥0, default: 1e-6

  // Robustness
  outlierThreshold?: number; // ≥0, default: 3.0
  adwinDelta?: number; // 0-1, default: 0.002
}
```

### Parameter Visual Guide

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         PARAMETER CATEGORIES                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🏗️ ARCHITECTURE          ⚡ OPTIMIZER           🛡️ ROBUSTNESS             │
│  ├─ hiddenLayers          ├─ learningRate       ├─ regularizationStrength │
│  ├─ convolutionsPerLayer  ├─ warmupSteps        ├─ convergenceThreshold   │
│  └─ kernelSize            ├─ totalSteps         ├─ outlierThreshold       │
│                           ├─ beta1              └─ adwinDelta             │
│                           ├─ beta2                                         │
│                           └─ epsilon                                       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Parameter Optimization Guide

### 🏗️ Architecture Parameters

#### `hiddenLayers` - Network Depth

Controls the number of convolutional layers in the network.

```
Complexity vs Depth Trade-off:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Layers │ Capacity   │ Training Speed │ Best For
───────┼────────────┼────────────────┼─────────────────────────────
  1    │ ▓░░░░░░░░░ │ ██████████     │ Simple linear relationships
  2    │ ▓▓▓░░░░░░░ │ ████████░░     │ Most general use cases ✓
  3-4  │ ▓▓▓▓▓░░░░░ │ ██████░░░░     │ Complex patterns
  5-7  │ ▓▓▓▓▓▓▓░░░ │ ████░░░░░░     │ Highly non-linear data
  8-10 │ ▓▓▓▓▓▓▓▓▓▓ │ ██░░░░░░░░     │ Very complex sequences
```

<details>
<summary><b>📌 Optimization Examples</b></summary>

**Simple Time Series (e.g., daily temperature):**

```typescript
const model = new ConvolutionalRegression({
  hiddenLayers: 1, // Simple pattern
  convolutionsPerLayer: 16,
});
```

**Financial Data (e.g., stock prices):**

```typescript
const model = new ConvolutionalRegression({
  hiddenLayers: 3, // Medium complexity
  convolutionsPerLayer: 64,
});
```

**Complex Multivariate Signals (e.g., sensor fusion):**

```typescript
const model = new ConvolutionalRegression({
  hiddenLayers: 5, // High complexity
  convolutionsPerLayer: 128,
});
```

</details>

---

#### `convolutionsPerLayer` - Network Width

Determines the number of filters (feature detectors) per convolutional layer.

```
Feature Extraction Capacity:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Filters │ Parameters │ Memory Usage │ Feature Diversity
────────┼────────────┼──────────────┼───────────────────
  8-16  │ Low        │ ~100KB       │ Basic patterns
  32    │ Medium     │ ~500KB       │ Standard use ✓
  64    │ High       │ ~2MB         │ Rich features
  128   │ Very High  │ ~8MB         │ Complex features
  256   │ Maximum    │ ~32MB        │ Full capacity
```

**Rule of Thumb:**

```
filters ≈ √(input_dimension × output_dimension) × complexity_factor

where complexity_factor:
  - Simple data: 1-2
  - Medium complexity: 2-4
  - Complex data: 4-8
```

<details>
<summary><b>📌 Code Examples</b></summary>

```typescript
// Low memory environment (embedded systems)
const lightModel = new ConvolutionalRegression({
  convolutionsPerLayer: 8,
  hiddenLayers: 1,
});

// Standard application
const standardModel = new ConvolutionalRegression({
  convolutionsPerLayer: 32, // Default
  hiddenLayers: 2,
});

// High-accuracy requirement
const accurateModel = new ConvolutionalRegression({
  convolutionsPerLayer: 128,
  hiddenLayers: 4,
});
```

</details>

---

#### `kernelSize` - Temporal Receptive Field

Controls how many adjacent input positions each filter examines.

```
Receptive Field Visualization:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:    [x₁] [x₂] [x₃] [x₄] [x₅] [x₆] [x₇] [x₈]

kernel=3:  └─┬──┘          Captures local patterns
             └──┘          (3 adjacent values)

kernel=5:  └───┬───┘       Captures medium-range patterns
               └───┘       (5 adjacent values)

kernel=7:  └─────┬─────┘   Captures long-range patterns
                 └─────┘   (7 adjacent values)
```

| Kernel Size | Pattern Type               | Use Case Example         |
| ----------- | -------------------------- | ------------------------ |
| `1`         | Point-wise transformations | Feature scaling          |
| `3`         | Short-term dependencies    | High-frequency signals ✓ |
| `5`         | Medium-term patterns       | Daily/weekly patterns    |
| `7`         | Long-term dependencies     | Seasonal trends          |
| `9+`        | Very long patterns         | Monthly/yearly cycles    |

<details>
<summary><b>📌 Selection Guide</b></summary>

```typescript
// High-frequency signal (millisecond samples)
const highFreqModel = new ConvolutionalRegression({
  kernelSize: 3, // Capture fast changes
  hiddenLayers: 2,
});

// Daily data with weekly patterns
const weeklyModel = new ConvolutionalRegression({
  kernelSize: 7, // Week = 7 days
  hiddenLayers: 3,
});

// Hourly data with daily patterns
const dailyModel = new ConvolutionalRegression({
  kernelSize: 5, // Capture ~5 hour windows
  hiddenLayers: 2,
});
```

**Pro Tip:** Use odd kernel sizes (3, 5, 7) for symmetric padding.

</details>

---

### ⚡ Optimizer Parameters

#### `learningRate` - Step Size

The most critical hyperparameter controlling update magnitude.

```
Learning Rate Spectrum:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

       1e-5        1e-4        1e-3        1e-2        1e-1
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
   ┌─────────┬───────────┬───────────┬───────────┬─────────┐
   │  Very   │  Fine     │  Default  │  Fast     │ Unstable│
   │  Slow   │  Tuning   │  ✓        │           │         │
   └─────────┴───────────┴───────────┴───────────┴─────────┘
   
   Convergence:  Slow ◀───────────────────────────────▶ Fast
   Stability:    High ◀───────────────────────────────▶ Low
```

**Learning Rate Selection Decision Tree:**

```
           ┌─────────────────────┐
           │ Is training stable? │
           └──────────┬──────────┘
                      │
     ┌────────────────┼────────────────┐
     ▼                ▼                ▼
  No/Diverging    Oscillating      Converging
     │                │                │
     ▼                ▼                ▼
Reduce by 10x    Reduce by 2-5x   Check speed
     │                │                │
     ▼                ▼                ▼
 lr × 0.1         lr × 0.3         Too slow?
                                       │
                           ┌───────────┴───────────┐
                           ▼                       ▼
                          Yes                      No
                           │                       │
                           ▼                       ▼
                     Increase by 2x            Keep lr ✓
```

<details>
<summary><b>📌 Practical Examples</b></summary>

```typescript
// Conservative approach (noisy data)
const conservativeModel = new ConvolutionalRegression({
  learningRate: 0.0001, // 10x smaller
  warmupSteps: 200, // Longer warmup
});

// Standard approach
const standardModel = new ConvolutionalRegression({
  learningRate: 0.001, // Default
  warmupSteps: 100,
});

// Aggressive approach (clean data, fast training)
const aggressiveModel = new ConvolutionalRegression({
  learningRate: 0.005, // 5x larger
  warmupSteps: 50, // Shorter warmup
});
```

**Adaptive Strategy:**

```typescript
// Start conservative, increase if stable
function adaptiveLearningRate(lossHistory: number[]): number {
  const recentLosses = lossHistory.slice(-10);
  const isStable = recentLosses.every((l, i) =>
    i === 0 || l <= recentLosses[i - 1] * 1.1
  );

  return isStable ? 0.002 : 0.0005;
}
```

</details>

---

#### `warmupSteps` & `totalSteps` - Learning Rate Schedule

Controls the learning rate progression over training.

```
Learning Rate Schedule Visualization:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LR │
   │     ╭────────╮
   │    ╱          ╲
   │   ╱            ╲
   │  ╱              ╲
   │ ╱                ╲
   │╱                  ╲_____________
   └──────────────────────────────────▶ Steps
   │◀─────▶│◀────────────────────────▶│
    Warmup      Cosine Decay Phase

Formula:
┌─────────────────────────────────────────────────────────────┐
│ Warmup (t ≤ warmupSteps):                                   │
│   lr(t) = learningRate × (t / warmupSteps)                  │
│                                                             │
│ Decay (t > warmupSteps):                                    │
│   progress = (t - warmupSteps) / (totalSteps - warmupSteps) │
│   lr(t) = learningRate × 0.5 × (1 + cos(π × progress))      │
└─────────────────────────────────────────────────────────────┘
```

<details>
<summary><b>📌 Schedule Configurations</b></summary>

```typescript
// Quick training (small dataset, <1000 samples)
const quickConfig = {
  warmupSteps: 50,
  totalSteps: 2000,
  learningRate: 0.002,
};

// Standard training (medium dataset, 1000-10000 samples)
const standardConfig = {
  warmupSteps: 100,
  totalSteps: 10000,
  learningRate: 0.001,
};

// Long training (large dataset, >10000 samples)
const longConfig = {
  warmupSteps: 500,
  totalSteps: 50000,
  learningRate: 0.0005,
};

// Streaming/continuous training
const streamingConfig = {
  warmupSteps: 100,
  totalSteps: 1000000, // Very long decay
  learningRate: 0.001,
};
```

</details>

---

#### `beta1` & `beta2` - Adam Momentum Parameters

Control the exponential moving averages in Adam optimizer.

```
Adam Update Visualization:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

             ┌────────────────────────────────────────┐
             │          ADAM OPTIMIZER                │
             │                                        │
  gradient   │  ┌─────────────────────────────────┐  │
      g  ───▶│  │ m = β₁·m + (1-β₁)·g             │  │  First moment
             │  │     (Momentum / Direction)       │  │  (β₁ = 0.9)
             │  └─────────────────────────────────┘  │
             │                 │                     │
             │                 ▼                     │
             │  ┌─────────────────────────────────┐  │
      g² ───▶│  │ v = β₂·v + (1-β₂)·g²            │  │  Second moment
             │  │     (Adaptive learning rate)     │  │  (β₂ = 0.999)
             │  └─────────────────────────────────┘  │
             │                 │                     │
             │                 ▼                     │
             │  ┌─────────────────────────────────┐  │
             │  │        m̂                         │  │
             │  │ Δw = ─────────                   │  │  Weight update
             │  │      √v̂ + ε                      │  │
             │  └─────────────────────────────────┘  │
             └────────────────────────────────────────┘
```

| Parameter | Default | Range        | Effect                                     |
| --------- | ------- | ------------ | ------------------------------------------ |
| `beta1`   | 0.9     | 0.8-0.99     | Higher = smoother gradients, more momentum |
| `beta2`   | 0.999   | 0.99-0.9999  | Higher = more stable per-parameter LR      |
| `epsilon` | 1e-8    | 1e-10 - 1e-6 | Prevents division by zero                  |

<details>
<summary><b>📌 When to Adjust</b></summary>

```typescript
// Noisy gradients (reduce momentum)
const noisyConfig = {
  beta1: 0.85, // Less momentum
  beta2: 0.999, // Keep stable
  learningRate: 0.0005,
};

// Sparse gradients (increase momentum)
const sparseConfig = {
  beta1: 0.95, // More momentum
  beta2: 0.9999, // Very stable scaling
  learningRate: 0.001,
};

// Default (works for most cases)
const defaultConfig = {
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
};
```

</details>

---

### 🛡️ Robustness Parameters

#### `regularizationStrength` - L2 Penalty

Prevents overfitting by penalizing large weights.

```
L2 Regularization Effect:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loss Function:
┌─────────────────────────────────────────────────────┐
│                                                     │
│   L_total = L_MSE + (λ/2) × Σ w²                   │
│             ─────   ──────────────                  │
│              ▲           ▲                          │
│              │           │                          │
│         Data fit    Weight penalty                  │
│                    (regularization)                 │
└─────────────────────────────────────────────────────┘

Effect on Weights:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

λ = 0 (no reg)     │██████████████████│  Large weights allowed
λ = 1e-5           │████████████░░░░░░│  Slight constraint
λ = 1e-4 (default) │██████████░░░░░░░░│  Balanced ✓
λ = 1e-3           │██████░░░░░░░░░░░░│  Strong constraint
λ = 1e-2           │████░░░░░░░░░░░░░░│  Very strong
```

<details>
<summary><b>📌 Selection Guide</b></summary>

```typescript
// Large dataset, low risk of overfitting
const largeDatsetConfig = {
  regularizationStrength: 1e-5, // Minimal regularization
};

// Standard dataset
const standardConfig = {
  regularizationStrength: 1e-4, // Default
};

// Small dataset, high overfitting risk
const smallDatasetConfig = {
  regularizationStrength: 1e-3, // Strong regularization
};

// Very small dataset (<100 samples)
const tinyDatasetConfig = {
  regularizationStrength: 5e-3, // Very strong
  hiddenLayers: 1, // Simpler model
  convolutionsPerLayer: 16,
};
```

**Validation Strategy:**

```typescript
function selectRegularization(trainLoss: number, valLoss: number): number {
  const overfitRatio = valLoss / trainLoss;

  if (overfitRatio > 2.0) return 1e-3; // High overfitting
  if (overfitRatio > 1.5) return 5e-4; // Moderate overfitting
  if (overfitRatio > 1.2) return 1e-4; // Slight overfitting
  return 1e-5; // Minimal overfitting
}
```

</details>

---

#### `outlierThreshold` - Anomaly Sensitivity

Z-score threshold for detecting and downweighting outliers.

```
Outlier Detection Mechanism:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    Normal Distribution of Errors
                           
                              ▲
                             ╱│╲
                            ╱ │ ╲
                           ╱  │  ╲
                          ╱   │   ╲
                         ╱    │    ╲
                        ╱     │     ╲
                    ▬▬▬▬▬▬▬▬▬▬│▬▬▬▬▬▬▬▬▬▬
               ────┼─────┼────┼────┼─────┼────▶
                  -3σ   -2σ   μ   +2σ   +3σ
                   │                     │
                   └─────────┬───────────┘
                             │
               outlierThreshold = 3.0 (default)
               
                   Points beyond ±3σ are outliers
                   and receive 0.1× weight
```

| Threshold | Coverage | False Positive Rate | Use Case                   |
| --------- | -------- | ------------------- | -------------------------- |
| `2.0`     | 95.4%    | High (4.6%)         | Aggressive outlier removal |
| `2.5`     | 98.8%    | Medium (1.2%)       | Moderate sensitivity       |
| `3.0`     | 99.7%    | Low (0.3%)          | Standard (default) ✓       |
| `3.5`     | 99.95%   | Very Low            | Conservative               |
| `4.0`     | 99.99%   | Minimal             | Only extreme outliers      |

<details>
<summary><b>📌 Configuration Examples</b></summary>

```typescript
// Clean data (minimal outliers expected)
const cleanDataConfig = {
  outlierThreshold: 4.0, // Only extreme cases
};

// Sensor data (occasional spikes)
const sensorConfig = {
  outlierThreshold: 3.0, // Default works well
};

// Financial data (frequent outliers)
const financialConfig = {
  outlierThreshold: 2.5, // More aggressive detection
};

// Noisy IoT data
const iotConfig = {
  outlierThreshold: 2.0, // Very aggressive
  regularizationStrength: 1e-3, // Also increase regularization
};
```

</details>

---

#### `adwinDelta` - Drift Detection Sensitivity

Controls the ADWIN algorithm's sensitivity to concept drift.

```
ADWIN Concept Drift Detection:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    Sliding Window
    ┌──────────────────────────────────────────┐
    │     W₀ (old data)    │   W₁ (new data)   │
    │    μ₀ = 0.05         │   μ₁ = 0.15       │
    └──────────────────────┼───────────────────┘
                           │
                     cut point

    Drift detected if: |μ₀ - μ₁| ≥ ε_cut
    
    where: ε_cut = √((1/2m) × ln(4|W|/δ))
    
    δ = adwinDelta (smaller = more sensitive)
```

```
Sensitivity Spectrum:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  δ = 0.1      δ = 0.01     δ = 0.002    δ = 0.0001
     │            │             │             │
     ▼            ▼             ▼             ▼
┌─────────┬───────────┬─────────────┬─────────────┐
│  Low    │  Medium   │   Default   │    High     │
│Sensitiv.│Sensitivity│     ✓       │ Sensitivity │
└─────────┴───────────┴─────────────┴─────────────┘

False Alarms:  Few ◀─────────────────────────▶ Many
Drift Detect:  Slow ◀────────────────────────▶ Fast
```

<details>
<summary><b>📌 Application-Specific Settings</b></summary>

```typescript
// Stable environment (rare drift)
const stableConfig = {
  adwinDelta: 0.01, // Low sensitivity
};

// Dynamic environment (frequent changes)
const dynamicConfig = {
  adwinDelta: 0.001, // High sensitivity
};

// Critical applications (immediate drift response)
const criticalConfig = {
  adwinDelta: 0.0001, // Very high sensitivity
  learningRate: 0.002, // Fast adaptation
};

// Monitoring drift without over-reacting
const monitoringConfig = {
  adwinDelta: 0.002, // Default, balanced
};
```

**Handling Drift Events:**

```typescript
const model = new ConvolutionalRegression({ adwinDelta: 0.002 });

function trainWithDriftHandling(data: FitInput) {
  const result = model.fitOnline(data);

  if (result.driftDetected) {
    console.log("⚠️ Concept drift detected!");
    // Option 1: Log and continue
    // Option 2: Increase learning rate temporarily
    // Option 3: Reset model for major drift
  }

  return result;
}
```

</details>

---

## 📊 Use Case Examples

### 📈 Time Series Forecasting

```typescript
/**
 * Stock Price Prediction Example
 * Features: [open, high, low, close, volume]
 * Target: [next_close]
 */
const stockModel = new ConvolutionalRegression({
  hiddenLayers: 3,
  convolutionsPerLayer: 64,
  kernelSize: 5, // Weekly patterns (5 trading days)
  learningRate: 0.0005, // Conservative for noisy data
  regularizationStrength: 1e-3,
  outlierThreshold: 2.5, // Financial data has outliers
});

// Training
for (const batch of stockDataBatches) {
  const result = stockModel.fitOnline({
    xCoordinates: batch.features,
    yCoordinates: batch.targets,
  });

  if (result.driftDetected) {
    console.log("📊 Market regime change detected");
  }
}

// Prediction with confidence intervals
const forecast = stockModel.predict(5); // 5-day forecast
forecast.predictions.forEach((pred, day) => {
  console.log(
    `Day ${day + 1}: $${pred.predicted[0].toFixed(2)} ` +
      `(95% CI: $${pred.lowerBound[0].toFixed(2)} - ` +
      `$${pred.upperBound[0].toFixed(2)})`,
  );
});
```

---

### 🌡️ Sensor Data Regression

```typescript
/**
 * Temperature Prediction from Multiple Sensors
 * Input: [sensor1, sensor2, sensor3, humidity, pressure]
 * Output: [temperature]
 */
const sensorModel = new ConvolutionalRegression({
  hiddenLayers: 2,
  convolutionsPerLayer: 32,
  kernelSize: 3,
  learningRate: 0.001,
  warmupSteps: 50,
  outlierThreshold: 3.0, // Handle sensor noise
});

// Continuous online learning
function processSensorReading(reading: SensorReading) {
  const result = sensorModel.fitOnline({
    xCoordinates: [reading.features],
    yCoordinates: [reading.temperature],
  });

  if (result.isOutlier) {
    console.warn("⚠️ Outlier detected - possible sensor malfunction");
  }

  return {
    loss: result.loss,
    prediction: sensorModel.predict(1).predictions[0],
  };
}
```

---

### 🤖 Real-time Control Systems

```typescript
/**
 * Robot Joint Position Prediction
 * Input: [joint_angles × 6, velocities × 6]
 * Output: [target_position × 3]
 */
const controlModel = new ConvolutionalRegression({
  hiddenLayers: 2,
  convolutionsPerLayer: 48,
  kernelSize: 3,
  learningRate: 0.002, // Fast adaptation
  warmupSteps: 20, // Quick warmup
  totalSteps: 5000,
  adwinDelta: 0.001, // Detect environmental changes
  convergenceThreshold: 1e-5,
});

// Real-time loop
async function controlLoop() {
  while (running) {
    const state = await getRobotState();

    // Update model with latest data
    const result = controlModel.fitOnline({
      xCoordinates: [state.input],
      yCoordinates: [state.targetPosition],
    });

    // Get next position prediction
    const prediction = controlModel.predict(1);

    if (prediction.isModelReady) {
      await sendCommand(prediction.predictions[0].predicted);
    }

    await sleep(10); // 100Hz control loop
  }
}
```

---

### 📊 Multi-Output Regression

```typescript
/**
 * Energy Consumption Forecasting
 * Input: [hour, dayOfWeek, month, temperature, humidity]
 * Output: [electricity, gas, water]
 */
const energyModel = new ConvolutionalRegression({
  hiddenLayers: 3,
  convolutionsPerLayer: 64,
  kernelSize: 7, // Weekly patterns
  learningRate: 0.001,
  regularizationStrength: 1e-4,
});

// Batch training
const history = [];
for (let epoch = 0; epoch < 10; epoch++) {
  const result = energyModel.fitOnline({
    xCoordinates: trainingFeatures,
    yCoordinates: trainingTargets,
  });

  history.push({
    epoch,
    loss: result.loss,
    accuracy: energyModel.getModelSummary().accuracy,
  });
}

// Multi-step forecast
const forecast = energyModel.predict(24); // 24-hour forecast
console.log("\n📊 24-Hour Energy Forecast:");
console.log("Hour | Electricity | Gas    | Water");
console.log("-----|-------------|--------|-------");
forecast.predictions.forEach((pred, hour) => {
  console.log(
    `${(hour + 1).toString().padStart(4)} | ` +
      `${pred.predicted[0].toFixed(2).padStart(11)} | ` +
      `${pred.predicted[1].toFixed(2).padStart(6)} | ` +
      `${pred.predicted[2].toFixed(2).padStart(5)}`,
  );
});
```

---

## 🧮 Mathematical Foundations

### Convolution Operation (Conv1D)

```
Same Padding Convolution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input x:   [x₀, x₁, x₂, x₃, x₄]    (spatial = 5)
Kernel w:  [w₀, w₁, w₂]            (kernelSize = 3)
Padding:   pad = (kernelSize - 1) / 2 = 1

Output y[i] = Σⱼ w[j] × x[i + j - pad]  (with zero-padding)

Example (i=0):
  y[0] = w[0]×0 + w[1]×x[0] + w[2]×x[1]
         (pad)

Example (i=2):
  y[2] = w[0]×x[1] + w[1]×x[2] + w[2]×x[3]
```

### Welford's Online Algorithm

```
Numerically Stable Running Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each new sample xₙ:

  1. δ = xₙ - μₙ₋₁           // Difference from current mean
  2. μₙ = μₙ₋₁ + δ/n         // Update mean
  3. δ₂ = xₙ - μₙ            // New difference
  4. M₂ₙ = M₂ₙ₋₁ + δ × δ₂    // Update sum of squared deviations

Final variance: σ² = M₂/(n-1)   // Bessel's correction
```

### Adam Optimizer

```
Adam Update Rule:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At timestep t:

  1. m_t = β₁ × m_{t-1} + (1 - β₁) × g_t     // First moment
  2. v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²    // Second moment
  
  3. m̂_t = m_t / (1 - β₁ᵗ)                   // Bias correction
  4. v̂_t = v_t / (1 - β₂ᵗ)                   // Bias correction
  
  5. θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)   // Update weights

With L2 regularization:
  g_t = ∇L(θ) + λ × θ                        // Add weight decay
```

### ADWIN Drift Detection

```
Adaptive Windowing Algorithm:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Window W with subwindows W₀, W₁:

  μ₀ = mean(W₀), μ₁ = mean(W₁)
  m = harmonic_mean(|W₀|, |W₁|)
  
  ε_cut = √((1/2m) × ln(4|W|/δ))
  
  Drift detected if: |μ₀ - μ₁| ≥ ε_cut
  
  On detection: discard W₀, continue with W₁
```

---

## 🎯 Best Practices

### ✅ Do's

```typescript
// ✅ Start with defaults, then tune
const model = new ConvolutionalRegression(); // Defaults are well-tuned

// ✅ Monitor training progress
const result = model.fitOnline(data);
if (result.loss > previousLoss * 2) {
  console.warn("Loss spike detected");
}

// ✅ Use model summary for debugging
const summary = model.getModelSummary();
console.log(`Accuracy: ${(summary.accuracy * 100).toFixed(2)}%`);

// ✅ Handle drift events
if (result.driftDetected) {
  // Log, adjust, or reset as needed
}

// ✅ Validate predictions
const pred = model.predict(1);
if (!pred.isModelReady) {
  console.warn("Model needs more training data");
}
```

### ❌ Don'ts

```typescript
// ❌ Don't use extreme learning rates
const bad1 = new ConvolutionalRegression({ learningRate: 1.0 }); // Too high!

// ❌ Don't skip warmup for new models
const bad2 = new ConvolutionalRegression({ warmupSteps: 0 }); // Unstable start

// ❌ Don't use too many layers for simple data
const bad3 = new ConvolutionalRegression({ hiddenLayers: 10 }); // Overkill

// ❌ Don't ignore outlier flags
const result = model.fitOnline(data);
// Always check: result.isOutlier

// ❌ Don't predict without sufficient training
const newModel = new ConvolutionalRegression();
newModel.predict(10); // isModelReady will be false!
```

---

## ⚠️ Troubleshooting

### Common Issues & Solutions

<details>
<summary><b>🔴 Loss is NaN or Infinite</b></summary>

**Causes:**

- Learning rate too high
- Input data contains NaN/Infinity
- Numerical overflow

**Solutions:**

```typescript
// Reduce learning rate
const model = new ConvolutionalRegression({
  learningRate: 0.0001, // 10x smaller
  epsilon: 1e-7, // Larger epsilon for stability
});

// Validate input data
function validateData(data: FitInput): boolean {
  for (const row of data.xCoordinates) {
    if (row.some((x) => !isFinite(x))) return false;
  }
  return true;
}
```

</details>

<details>
<summary><b>🟡 Loss Not Decreasing</b></summary>

**Causes:**

- Learning rate too low
- Model too simple for data
- Data not properly formatted

**Solutions:**

```typescript
// Increase learning rate
const model = new ConvolutionalRegression({
  learningRate: 0.005, // Increase
  warmupSteps: 50, // Shorter warmup
});

// Or increase model capacity
const biggerModel = new ConvolutionalRegression({
  hiddenLayers: 4,
  convolutionsPerLayer: 128,
});
```

</details>

<details>
<summary><b>🟡 High Variance in Predictions</b></summary>

**Causes:**

- Insufficient training data
- High noise in data
- Model overfitting

**Solutions:**

```typescript
const model = new ConvolutionalRegression({
  regularizationStrength: 1e-3, // Increase regularization
  outlierThreshold: 2.5, // More aggressive outlier handling
  hiddenLayers: 1, // Simpler model
});
```

</details>

<details>
<summary><b>🟢 Frequent Drift Detection</b></summary>

**Causes:**

- adwinDelta too small
- Legitimately changing data distribution

**Solutions:**

```typescript
// If false positives:
const model = new ConvolutionalRegression({
  adwinDelta: 0.01, // Less sensitive
});

// If legitimate drift - embrace it:
function handleDrift(result: FitResult) {
  if (result.driftDetected) {
    // Drift is expected, model adapts automatically
    console.log("Distribution shift detected and handled");
  }
}
```

</details>

---

## 📈 Performance Tips

### Memory Optimization

```typescript
// Use smaller model for memory-constrained environments
const lightweightModel = new ConvolutionalRegression({
  hiddenLayers: 1,
  convolutionsPerLayer: 16,
  kernelSize: 3,
});

// Approximate memory usage:
// Parameters ≈ hiddenLayers × convolutionsPerLayer² × kernelSize × 8 bytes
// Example: 2 × 32² × 3 × 8 = ~49KB for weights alone
```

### Training Speed Optimization

```typescript
// Batch processing for speed
const BATCH_SIZE = 32;

for (let i = 0; i < data.length; i += BATCH_SIZE) {
  const batch = {
    xCoordinates: data.xCoordinates.slice(i, i + BATCH_SIZE),
    yCoordinates: data.yCoordinates.slice(i, i + BATCH_SIZE),
  };
  model.fitOnline(batch);
}
```

### Prediction Performance

```typescript
// Cache predictions when possible
const predictionCache = new Map<string, PredictionResult>();

function getCachedPrediction(
  model: ConvolutionalRegression,
  steps: number,
): PredictionResult {
  const key = `${model.getModelSummary().sampleCount}-${steps}`;

  if (!predictionCache.has(key)) {
    predictionCache.set(key, model.predict(steps));
  }

  return predictionCache.get(key)!;
}
```

---

## 📜 License

MIT © 2025 Henrique Emanoel Viana

---

<div align="center">

**Built with ❤️ for the machine learning community**

[⬆ Back to Top](#-convolutionalregression)

</div>
