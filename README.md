Model: # 🧠 ConvolutionalRegression

<div align="center">

**A High-Performance 1D Convolutional Neural Network for Multivariate Regression
with Online Learning**

[Features](#-features) • [Installation](#-installation) •
[Quick Start](#-quick-start) • [API Reference](#-api-reference) •
[Parameter Guide](#-parameter-optimization-guide) • [Examples](#-examples)

---

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│ 📊 Input → [🔲 Conv1D → ⚡ ReLU]×L → 📐 Flatten → 🔗 Dense → 📈 Output │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Architecture Overview](#-architecture-overview)
- [API Reference](#-api-reference)
- [Parameter Optimization Guide](#-parameter-optimization-guide)
- [Examples](#-examples)
- [Algorithms & Concepts](#-algorithms--concepts)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Capabilities

| Feature                       | Description                                |
| ----------------------------- | ------------------------------------------ |
| 🔄 **Online Learning**        | Incremental training, one sample at a time |
| 📦 **Batch Learning**         | Traditional mini-batch gradient descent    |
| 🎲 **Multivariate I/O**       | Multiple input & output dimensions         |
| 🔮 **Uncertainty Estimation** | Confidence bounds on predictions           |
| 🚨 **Drift Detection**        | ADWIN algorithm for concept drift          |
| 🛡️ **Outlier Handling**       | Z-score based outlier downweighting        |

</td>
<td width="50%">

### ⚡ Performance Features

| Feature                    | Description                           |
| -------------------------- | ------------------------------------- |
| 🏎️ **Float64Arrays**       | High-precision typed arrays           |
| 🧹 **Zero GC Pressure**    | Preallocated buffers throughout       |
| 📐 **In-Place Operations** | Memory-efficient computations         |
| 🎛️ **Adam Optimizer**      | Adaptive learning rates per parameter |
| 📈 **LR Scheduling**       | Linear warmup + cosine decay          |
| 🔧 **L2 Regularization**   | Weight decay for generalization       |

</td>
</tr>
</table>

---

### Direct Import

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";
```

---

## 🚀 Quick Start

### Basic Usage

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// 1️⃣ Create model with default settings
const model = new ConvolutionalRegression();

// 2️⃣ Online Learning - Train incrementally
for (const sample of dataStream) {
  const result = model.fitOnline({
    xCoordinates: [sample.features],
    yCoordinates: [sample.targets],
  });

  console.log(
    `Loss: ${result.loss.toFixed(4)}, Converged: ${result.converged}`,
  );
}

// 3️⃣ Make Predictions
const predictions = model.predict(5); // Predict 5 future steps

for (const pred of predictions.predictions) {
  console.log(`Predicted: ${pred.predicted}`);
  console.log(`95% CI: [${pred.lowerBound}, ${pred.upperBound}]`);
}
```

### Batch Training

```typescript
// Prepare your dataset
const trainX = [[1, 2, 3], [2, 3, 4], [3, 4, 5] /* ... */];
const trainY = [[4, 5], [5, 6], [6, 7] /* ... */];

// Create model with custom configuration
const model = new ConvolutionalRegression({
  hiddenLayers: 3,
  convolutionsPerLayer: 64,
  learningRate: 0.001,
});

// Train in batches
const result = model.fitBatch({
  xCoordinates: trainX,
  yCoordinates: trainY,
  epochs: 100,
});

console.log(`Final Loss: ${result.finalLoss}`);
console.log(
  `Converged: ${result.converged} after ${result.epochsCompleted} epochs`,
);
```

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CONVOLUTIONAL REGRESSION NETWORK                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                             │
│  │   INPUT     │  Shape: [1, inputDim]                                       │
│  │  (Raw Data) │  • Z-score normalized using Welford's algorithm             │
│  └──────┬──────┘                                                             │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CONVOLUTIONAL BLOCKS × L                          │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │    │
│  │  │  Conv1D     │    │    ReLU     │    │   Output    │              │    │
│  │  │  (same pad) │ ──▶│  Activation │ ──▶│  [C, W]     │              │    │
│  │  │  K filters  │    │  max(0, x)  │    │             │              │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │    │
│  │                                                                      │    │
│  │  Layer 1: 1 → C channels    Layer 2-L: C → C channels               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐                                                             │
│  │   FLATTEN   │  Shape: [C × inputDim]                                      │
│  └──────┬──────┘                                                             │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐                                                             │
│  │    DENSE    │  Fully connected: [C × inputDim] → [outputDim]              │
│  │   (Linear)  │  y = Wx + b                                                 │
│  └──────┬──────┘                                                             │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐                                                             │
│  │   OUTPUT    │  Shape: [outputDim]                                         │
│  │(Predictions)│  • Denormalized to original scale                           │
│  └─────────────┘  • Uncertainty bounds computed                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Where:
  L = hiddenLayers (default: 2)
  C = convolutionsPerLayer (default: 32)
  K = kernelSize (default: 3)
```

### Data Flow Diagram

```
                       TRAINING FLOW
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Raw Input (x)              Raw Target (y)                  │
│        │                          │                          │
│        ▼                          ▼                          │
│  ┌──────────┐              ┌──────────┐                      │
│  │ Normalize│              │ Normalize│  (Welford's)         │
│  └────┬─────┘              └────┬─────┘                      │
│       │                         │                            │
│       ▼                         │                            │
│  ┌──────────┐                   │                            │
│  │ Forward  │                   │                            │
│  │  Pass    │                   │                            │
│  └────┬─────┘                   │                            │
│       │                         │                            │
│       ▼                         ▼                            │
│  ┌────────────────────────────────┐                          │
│  │     Compute MSE Loss           │                          │
│  │   L = ½‖ŷ - y‖² + λ‖W‖²       │                          │
│  └─────────────┬──────────────────┘                          │
│                │                                             │
│                ▼                                             │
│  ┌──────────────────────────────────────┐                    │
│  │        Backward Pass                  │                    │
│  │   • Compute gradients                 │                    │
│  │   • Adam optimizer update             │                    │
│  │   • Apply L2 regularization           │                    │
│  └──────────────────────────────────────┘                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📚 API Reference

### Constructor

```typescript
const model = new ConvolutionalRegression(config?: ConvolutionalRegressionConfig);
```

### Configuration Interface

```typescript
interface ConvolutionalRegressionConfig {
  hiddenLayers?: number; // 1-10, default: 2
  convolutionsPerLayer?: number; // 1-256, default: 32
  kernelSize?: number; // default: 3
  learningRate?: number; // default: 0.001
  warmupSteps?: number; // default: 100
  totalSteps?: number; // default: 10000
  beta1?: number; // default: 0.9
  beta2?: number; // default: 0.999
  epsilon?: number; // default: 1e-8
  regularizationStrength?: number; // default: 1e-4
  batchSize?: number; // default: 32
  convergenceThreshold?: number; // default: 1e-6
  outlierThreshold?: number; // default: 3.0
  adwinDelta?: number; // default: 0.002
}
```

---

### Methods

#### 🔄 `fitOnline(data)`

Performs one step of online (incremental) learning.

```typescript
fitOnline(data: {
  xCoordinates: number[][];
  yCoordinates: number[][];
}): FitResult
```

**Parameters:**

| Parameter      | Type         | Description                         |
| -------------- | ------------ | ----------------------------------- |
| `xCoordinates` | `number[][]` | Input features (uses first element) |
| `yCoordinates` | `number[][]` | Target values (uses first element)  |

**Returns: `FitResult`**

| Property                | Type      | Description                               |
| ----------------------- | --------- | ----------------------------------------- |
| `loss`                  | `number`  | Current loss value (MSE + regularization) |
| `gradientNorm`          | `number`  | L2 norm of the gradient                   |
| `effectiveLearningRate` | `number`  | Learning rate after warmup/decay          |
| `isOutlier`             | `boolean` | Whether sample was detected as outlier    |
| `converged`             | `boolean` | Whether model has converged               |
| `sampleIndex`           | `number`  | Number of samples processed               |
| `driftDetected`         | `boolean` | Whether concept drift was detected        |

**Example:**

```typescript
const result = model.fitOnline({
  xCoordinates: [[1.0, 2.0, 3.0, 4.0, 5.0]],
  yCoordinates: [[6.0, 7.0]],
});

if (result.driftDetected) {
  console.log("⚠️ Concept drift detected! Model adapting...");
}

if (result.isOutlier) {
  console.log("🔍 Outlier detected, downweighted in training");
}
```

---

#### 📦 `fitBatch(data)`

Performs batch training with mini-batch gradient descent.

```typescript
fitBatch(data: {
  xCoordinates: number[][];
  yCoordinates: number[][];
  epochs?: number;
}): BatchFitResult
```

**Parameters:**

| Parameter      | Type         | Default  | Description             |
| -------------- | ------------ | -------- | ----------------------- |
| `xCoordinates` | `number[][]` | required | Array of input samples  |
| `yCoordinates` | `number[][]` | required | Array of target values  |
| `epochs`       | `number`     | `100`    | Maximum training epochs |

**Returns: `BatchFitResult`**

| Property                | Type       | Description                      |
| ----------------------- | ---------- | -------------------------------- |
| `finalLoss`             | `number`   | Final loss after training        |
| `lossHistory`           | `number[]` | Loss value per epoch             |
| `converged`             | `boolean`  | Whether training converged early |
| `epochsCompleted`       | `number`   | Number of epochs completed       |
| `totalSamplesProcessed` | `number`   | Total samples across all epochs  |

**Example:**

```typescript
const result = model.fitBatch({
  xCoordinates: trainX,
  yCoordinates: trainY,
  epochs: 200,
});

// Plot loss curve
result.lossHistory.forEach((loss, epoch) => {
  console.log(`Epoch ${epoch + 1}: Loss = ${loss.toFixed(6)}`);
});

if (result.converged) {
  console.log(`✅ Converged after ${result.epochsCompleted} epochs`);
}
```

---

#### 🔮 `predict(futureSteps)`

Makes predictions for future steps with uncertainty estimates.

```typescript
predict(futureSteps: number): PredictionResult
```

**Parameters:**

| Parameter     | Type     | Description                       |
| ------------- | -------- | --------------------------------- |
| `futureSteps` | `number` | Number of predictions to generate |

**Returns: `PredictionResult`**

| Property       | Type                 | Description                              |
| -------------- | -------------------- | ---------------------------------------- |
| `predictions`  | `SinglePrediction[]` | Array of predictions                     |
| `accuracy`     | `number`             | Model accuracy estimate: 1/(1 + avgLoss) |
| `sampleCount`  | `number`             | Number of training samples seen          |
| `isModelReady` | `boolean`            | Whether model has been trained           |

**`SinglePrediction` Structure:**

| Property        | Type       | Description                          |
| --------------- | ---------- | ------------------------------------ |
| `predicted`     | `number[]` | Predicted output values              |
| `lowerBound`    | `number[]` | Lower confidence bound (mean - 2×SE) |
| `upperBound`    | `number[]` | Upper confidence bound (mean + 2×SE) |
| `standardError` | `number[]` | Standard error per dimension         |

**Example:**

```typescript
const result = model.predict(10);

if (!result.isModelReady) {
  console.log("⚠️ Model needs training first!");
  return;
}

console.log(`Model accuracy: ${(result.accuracy * 100).toFixed(2)}%`);

result.predictions.forEach((pred, step) => {
  console.log(`\n📊 Step ${step + 1}:`);
  pred.predicted.forEach((val, dim) => {
    console.log(`  Dimension ${dim}: ${val.toFixed(4)}`);
    console.log(
      `    95% CI: [${pred.lowerBound[dim].toFixed(4)}, ${
        pred.upperBound[dim].toFixed(4)
      }]`,
    );
    console.log(`    SE: ±${pred.standardError[dim].toFixed(4)}`);
  });
});
```

---

#### 📊 `getModelSummary()`

Returns comprehensive model information.

```typescript
getModelSummary(): ModelSummary
```

**Returns: `ModelSummary`**

| Property                | Type      | Description                        |
| ----------------------- | --------- | ---------------------------------- |
| `isInitialized`         | `boolean` | Whether model has been initialized |
| `inputDimension`        | `number`  | Input feature dimension            |
| `outputDimension`       | `number`  | Output dimension                   |
| `hiddenLayers`          | `number`  | Number of convolutional layers     |
| `convolutionsPerLayer`  | `number`  | Filters per layer                  |
| `kernelSize`            | `number`  | Convolution kernel size            |
| `totalParameters`       | `number`  | Total trainable parameters         |
| `sampleCount`           | `number`  | Training samples seen              |
| `accuracy`              | `number`  | Current accuracy estimate          |
| `converged`             | `boolean` | Whether model has converged        |
| `effectiveLearningRate` | `number`  | Current learning rate              |
| `driftCount`            | `number`  | Number of drift events detected    |

**Example:**

```typescript
const summary = model.getModelSummary();

console.log("╔════════════════════════════════════════╗");
console.log("║         MODEL SUMMARY                  ║");
console.log("╠════════════════════════════════════════╣");
console.log(
  `║ Architecture: ${summary.inputDimension} → ${summary.hiddenLayers}×Conv → ${summary.outputDimension}`,
);
console.log(`║ Parameters: ${summary.totalParameters.toLocaleString()}`);
console.log(`║ Samples Trained: ${summary.sampleCount.toLocaleString()}`);
console.log(`║ Accuracy: ${(summary.accuracy * 100).toFixed(2)}%`);
console.log(
  `║ Learning Rate: ${summary.effectiveLearningRate.toExponential(3)}`,
);
console.log(`║ Drift Events: ${summary.driftCount}`);
console.log("╚════════════════════════════════════════╝");
```

---

#### 🔧 `getWeights()` / `getNormalizationStats()` / `reset()`

```typescript
// Get all weights and optimizer state
getWeights(): WeightInfo

// Get normalization statistics
getNormalizationStats(): NormalizationStats

// Reset model to initial state
reset(): void
```

---

## 🎯 Parameter Optimization Guide

### Quick Reference Table

| Use Case                   | Hidden Layers | Conv/Layer | Kernel | Learning Rate | Batch Size |
| -------------------------- | :-----------: | :--------: | :----: | :-----------: | :--------: |
| 🏃 Real-time streaming     |      1-2      |   16-32    |   3    |  0.001-0.01   | 1 (online) |
| 📊 Time series forecasting |      2-3      |   32-64    |  3-5   | 0.0005-0.001  |   32-64    |
| 🔬 Complex patterns        |      3-5      |   64-128   |  5-7   | 0.0001-0.0005 |   64-128   |
| 💾 Limited memory          |       1       |    8-16    |   3    |     0.001     |     16     |
| 🎯 High precision          |      4-6      |  128-256   |   5    |    0.0001     |     32     |

---

### 📐 `hiddenLayers` (1-10, default: 2)

**Controls network depth - the number of convolutional blocks.**

```
Depth = 1:    Input → [Conv→ReLU] → Dense → Output
Depth = 2:    Input → [Conv→ReLU] → [Conv→ReLU] → Dense → Output
Depth = 3:    Input → [Conv→ReLU]³ → Dense → Output
```

| Value   | Best For                               | Trade-offs                  |
| ------- | -------------------------------------- | --------------------------- |
| **1**   | Simple linear patterns, fast inference | Limited pattern complexity  |
| **2**   | Most general use cases                 | Good balance                |
| **3-4** | Multi-scale temporal patterns          | Increased training time     |
| **5+**  | Very complex hierarchical patterns     | Risk of vanishing gradients |

**Example: Choosing Depth Based on Pattern Complexity**

```typescript
// Simple trend detection
const simpleModel = new ConvolutionalRegression({
  hiddenLayers: 1,
  convolutionsPerLayer: 16,
});

// Complex seasonal + trend patterns
const complexModel = new ConvolutionalRegression({
  hiddenLayers: 4,
  convolutionsPerLayer: 64,
  learningRate: 0.0003, // Lower LR for deeper networks
});
```

**📊 Receptive Field Formula:**

```
Receptive Field = 1 + hiddenLayers × (kernelSize - 1)

Example with kernelSize=3:
  1 layer  → RF = 3  (sees 3 time steps)
  2 layers → RF = 5  (sees 5 time steps)
  3 layers → RF = 7  (sees 7 time steps)
```

---

### 🔲 `convolutionsPerLayer` (1-256, default: 32)

**Controls network width - the number of feature maps per layer.**

```
                 ┌──────────────────────────────────────┐
                 │        Feature Map Visualization      │
                 ├──────────────────────────────────────┤
  8 filters:     │  ████  ████  ████  ████              │  (fast, limited)
 32 filters:     │  ████████████████████████████████    │  (balanced)
128 filters:     │  ████████████████████████████████    │  (powerful, slow)
                 │  ████████████████████████████████    │
                 │  ████████████████████████████████    │
                 │  ████████████████████████████████    │
                 └──────────────────────────────────────┘
```

| Value       | Parameters | Memory  | Speed  | Capacity  |
| ----------- | ---------- | ------- | ------ | --------- |
| **8-16**    | Low        | ~KB     | ⚡⚡⚡ | Basic     |
| **32**      | Medium     | ~10KB   | ⚡⚡   | Good      |
| **64**      | High       | ~50KB   | ⚡     | Very Good |
| **128-256** | Very High  | ~200KB+ | 🐢     | Maximum   |

**Example: Scaling for Dataset Size**

```typescript
// Small dataset (< 1,000 samples)
const smallDataModel = new ConvolutionalRegression({
  convolutionsPerLayer: 16, // Prevent overfitting
  regularizationStrength: 1e-3,
});

// Large dataset (> 100,000 samples)
const largeDataModel = new ConvolutionalRegression({
  convolutionsPerLayer: 128,
  regularizationStrength: 1e-5,
});
```

---

### 📏 `kernelSize` (default: 3)

**Controls the local receptive field of each convolution.**

```
kernelSize = 3:    [ . ][███][███][███][ . ][ . ]
                         ↓
                       output

kernelSize = 5:    [ . ][███][███][███][███][███][ . ]
                              ↓
                            output

kernelSize = 7:    [███][███][███][███][███][███][███]
                                 ↓
                               output
```

| Value          | Best For                             | Notes                |
| -------------- | ------------------------------------ | -------------------- |
| **3**          | Most time series, local patterns     | Fast, efficient      |
| **5**          | Weekly patterns (daily data)         | Good for periodicity |
| **7**          | Longer-range dependencies            | More parameters      |
| **Odd values** | Always use odd for symmetric padding | Required             |

**Example: Matching Kernel to Seasonality**

```typescript
// Hourly data with daily patterns (24 hours)
const hourlyModel = new ConvolutionalRegression({
  kernelSize: 5,
  hiddenLayers: 4, // 4 layers × (5-1) + 1 = 17 receptive field
});

// Daily data with weekly patterns (7 days)
const dailyModel = new ConvolutionalRegression({
  kernelSize: 7,
  hiddenLayers: 2,
});
```

---

### 📈 `learningRate` (default: 0.001)

**Base learning rate for Adam optimizer, subject to warmup and decay.**

```
Learning Rate Schedule:
                                                     
    LR │     ╭──────────╮                            
       │    ╱            ╲                           
       │   ╱              ╲                          
       │  ╱                ╲                         
       │ ╱                  ╲                        
       │╱                    ╲____                   
       └────────────────────────────► Steps         
        │← Warmup →│← Cosine Decay →│
```

**Learning Rate Schedule Formula:**

```
During warmup (t < warmupSteps):
    η(t) = baseLR × (t + 1) / warmupSteps

After warmup:
    progress = (t - warmupSteps) / (totalSteps - warmupSteps)
    η(t) = baseLR × 0.5 × (1 + cos(π × progress))
```

| Value       | Stability | Convergence     | Use Case                   |
| ----------- | --------- | --------------- | -------------------------- |
| **0.01**    | Low       | Fast but risky  | Quick experiments          |
| **0.001**   | Medium    | Balanced        | Default choice             |
| **0.0001**  | High      | Slow but stable | Fine-tuning, deep networks |
| **0.00001** | Very High | Very slow       | Precise convergence        |

**Example: Learning Rate Selection Strategy**

```typescript
// Fast prototyping
const prototypeModel = new ConvolutionalRegression({
  learningRate: 0.01,
  warmupSteps: 50,
  totalSteps: 1000,
});

// Production model with stability
const productionModel = new ConvolutionalRegression({
  learningRate: 0.0005,
  warmupSteps: 200,
  totalSteps: 50000,
});

// Fine-tuning pre-trained knowledge
const fineTuneModel = new ConvolutionalRegression({
  learningRate: 0.00005,
  warmupSteps: 100,
});
```

---

### 🔥 `warmupSteps` (default: 100)

**Number of steps to linearly increase learning rate from 0 to base.**

```
Warmup Effect on Training Stability:

Without warmup:          With warmup:
    │                        │
Loss│ ╭╮                Loss │ 
    │╱  ╲                    │   ╲
    │    ╲                   │    ╲
    │     ╲                  │     ╲
    │      ──────            │      ──────
    └────────────►           └────────────►
      Steps                    Steps

(Oscillations)            (Smooth descent)
```

| Value        | When to Use                     |
| ------------ | ------------------------------- |
| **0**        | Very small LR, stable gradients |
| **50-100**   | Small datasets, online learning |
| **100-500**  | Standard batch training         |
| **500-1000** | Large batches, high LR          |

**Example: Warmup for Different Scenarios**

```typescript
// Online learning (single samples)
const onlineModel = new ConvolutionalRegression({
  warmupSteps: 50, // Quick warmup
  learningRate: 0.001,
});

// Large batch training
const batchModel = new ConvolutionalRegression({
  warmupSteps: 500,
  batchSize: 128,
  learningRate: 0.002, // Higher LR with warmup
});
```

---

### 🛡️ `regularizationStrength` (L2, default: 1e-4)

**Weight decay coefficient to prevent overfitting.**

```
L2 Regularization Loss:
    L_total = L_data + (λ/2) × Σ||W||²

Effect on Weights:
    ┌─────────────────────────────────┐
    │ λ = 0:     Weights can grow     │
    │            unbounded            │
    │                                 │
    │ λ = 1e-4:  Mild constraint      │
    │            (default)            │
    │                                 │
    │ λ = 1e-2:  Strong constraint    │
    │            (smaller weights)    │
    └─────────────────────────────────┘
```

| Value    | Effect            | Best For                        |
| -------- | ----------------- | ------------------------------- |
| **0**    | No regularization | Large datasets                  |
| **1e-5** | Very weak         | Large, clean datasets           |
| **1e-4** | Weak (default)    | Most cases                      |
| **1e-3** | Medium            | Small datasets, overfitting     |
| **1e-2** | Strong            | Very small datasets, noisy data |

**Example: Regularization for Dataset Size**

```typescript
// Large dataset - minimal regularization
const largeDataset = new ConvolutionalRegression({
  regularizationStrength: 1e-5,
  convolutionsPerLayer: 128,
});

// Small dataset - strong regularization
const smallDataset = new ConvolutionalRegression({
  regularizationStrength: 1e-3,
  convolutionsPerLayer: 32,
});

// Noisy data - extra regularization
const noisyData = new ConvolutionalRegression({
  regularizationStrength: 5e-3,
  outlierThreshold: 2.5, // More aggressive outlier detection
});
```

---

### 🎯 `outlierThreshold` (default: 3.0)

**Z-score threshold for outlier detection and downweighting.**

```
Outlier Detection:
    z = |residual - mean| / std
    
    If z > threshold → outlier → weight = 0.1
    Otherwise        → normal  → weight = 1.0

                    Normal Distribution
                         ╭────╮
                       ╭─╯    ╰─╮
                     ╭─╯        ╰─╮
                   ╭─╯            ╰─╮
    ──────────────╯                ╰──────────────
                  │      │      │
               -3σ     mean    +3σ
                  │←──────────────→│
                   99.7% of data
                        │
               Outliers detected outside
```

| Value   | Sensitivity | Downweighted | Use Case               |
| ------- | ----------- | ------------ | ---------------------- |
| **2.0** | High        | ~5%          | Very clean data        |
| **2.5** | Medium-High | ~1.2%        | Clean data             |
| **3.0** | Medium      | ~0.3%        | Default                |
| **4.0** | Low         | ~0.006%      | Noisy/variable data    |
| **5.0** | Very Low    | ~0.00006%    | Keep almost everything |

**Example: Outlier Handling Strategies**

```typescript
// Sensor data (occasional spikes)
const sensorModel = new ConvolutionalRegression({
  outlierThreshold: 2.5, // More aggressive outlier detection
});

// Financial data (fat tails expected)
const financialModel = new ConvolutionalRegression({
  outlierThreshold: 4.0, // Keep more extreme values
});

// Monitor outliers in training
const result = model.fitOnline({ xCoordinates: [x], yCoordinates: [y] });
if (result.isOutlier) {
  console.log("📍 Outlier detected, weight reduced to 0.1");
}
```

---

### 🌊 `adwinDelta` (default: 0.002)

**ADWIN confidence parameter for concept drift detection.**

```
ADWIN Algorithm:
    ┌─────────────────────────────────────────────────┐
    │                Sliding Window                    │
    │  ┌──────────────────┬──────────────────┐        │
    │  │    Window 1      │    Window 2      │        │
    │  │    (older)       │    (newer)       │        │
    │  │    mean = μ₁     │    mean = μ₂     │        │
    │  └──────────────────┴──────────────────┘        │
    │                                                  │
    │  If |μ₁ - μ₂| > ε_cut  →  DRIFT DETECTED!       │
    │                                                  │
    │  ε_cut = √((1/2m) × ln(4/δ))                    │
    │                                                  │
    │  Smaller δ → larger ε_cut → fewer false alarms  │
    └─────────────────────────────────────────────────┘
```

| Value      | Sensitivity | False Positives | Response Time |
| ---------- | ----------- | --------------- | ------------- |
| **0.01**   | Low         | Very few        | Slow          |
| **0.002**  | Medium      | Few (default)   | Moderate      |
| **0.0005** | High        | Some            | Fast          |
| **0.0001** | Very High   | Many            | Very Fast     |

**Example: Drift Detection Configuration**

```typescript
// Stable environment - rare changes
const stableModel = new ConvolutionalRegression({
  adwinDelta: 0.01, // Low sensitivity
});

// Dynamic environment - frequent changes
const dynamicModel = new ConvolutionalRegression({
  adwinDelta: 0.0005, // High sensitivity
});

// Monitor drift in production
const result = model.fitOnline({ xCoordinates: [x], yCoordinates: [y] });
if (result.driftDetected) {
  console.log("🌊 Concept drift detected!");
  console.log("Model is adapting to new data distribution...");

  const summary = model.getModelSummary();
  console.log(`Total drift events: ${summary.driftCount}`);
}
```

---

### 📦 `batchSize` (default: 32)

**Mini-batch size for batch training.**

```
Batch Size Effects:
                                                     
    ┌──────────────────────────────────────────────┐
    │  Small Batch (8-16)    │  Large Batch (128+) │
    ├────────────────────────┼─────────────────────┤
    │  ✓ Better generalize   │  ✓ Faster compute   │
    │  ✓ Escape local minima │  ✓ Stable gradients │
    │  ✗ Noisy gradients     │  ✗ May overfit      │
    │  ✗ Slower convergence  │  ✗ More memory      │
    └────────────────────────┴─────────────────────┘
```

| Value      | Memory    | GPU Efficiency | Generalization |
| ---------- | --------- | -------------- | -------------- |
| **8-16**   | Low       | Low            | Excellent      |
| **32**     | Medium    | Medium         | Very Good      |
| **64-128** | High      | High           | Good           |
| **256+**   | Very High | Very High      | Fair           |

**Example: Batch Size Selection**

```typescript
// Memory-constrained environment
const lowMemModel = new ConvolutionalRegression({
  batchSize: 16,
  learningRate: 0.0005, // Lower LR for smaller batches
});

// High-performance training
const highPerfModel = new ConvolutionalRegression({
  batchSize: 128,
  learningRate: 0.002, // Higher LR for larger batches
  warmupSteps: 500,
});
```

---

### ⚖️ Adam Optimizer Parameters

#### `beta1` (default: 0.9) - First Moment Decay

```
First moment (momentum):  m = β₁·m + (1-β₁)·g
    
    β₁ = 0.9:   Current gradient has 10% influence
    β₁ = 0.99:  Current gradient has 1% influence (more momentum)
```

#### `beta2` (default: 0.999) - Second Moment Decay

```
Second moment (adaptive LR):  v = β₂·v + (1-β₂)·g²
    
    β₂ = 0.999:  Long-term variance tracking (stable)
    β₂ = 0.99:   Faster adaptation (more responsive)
```

#### `epsilon` (default: 1e-8) - Numerical Stability

```
Update rule:  W -= η · m̂ / (√v̂ + ε)
    
    ε prevents division by zero when gradients are very small
```

**Example: Fine-tuning Adam**

```typescript
// Sparse gradients (some features rarely update)
const sparseModel = new ConvolutionalRegression({
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-7, // Slightly larger for stability
});

// Rapidly changing gradients
const dynamicModel = new ConvolutionalRegression({
  beta1: 0.85, // Less momentum
  beta2: 0.99, // Faster adaptation
});
```

---

## 📖 Examples

### Example 1: Time Series Forecasting

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// Generate synthetic time series with trend and seasonality
function generateTimeSeries(n: number): { x: number[][]; y: number[][] } {
  const x: number[][] = [];
  const y: number[][] = [];

  for (let i = 0; i < n; i++) {
    const t = i * 0.1;
    const trend = 0.01 * t;
    const seasonal = Math.sin(t) + 0.5 * Math.cos(2 * t);
    const noise = (Math.random() - 0.5) * 0.1;

    // Input: last 10 values
    const input: number[] = [];
    for (let j = 0; j < 10; j++) {
      const tj = (i - 10 + j) * 0.1;
      input.push(0.01 * tj + Math.sin(tj) + 0.5 * Math.cos(2 * tj));
    }

    // Output: next 2 values
    const output = [trend + seasonal + noise, trend + seasonal + 0.1 + noise];

    x.push(input);
    y.push(output);
  }

  return { x, y };
}

// Create and train model
const model = new ConvolutionalRegression({
  hiddenLayers: 3,
  convolutionsPerLayer: 64,
  kernelSize: 5,
  learningRate: 0.001,
  batchSize: 32,
});

const data = generateTimeSeries(1000);

console.log("🚀 Starting training...\n");

const result = model.fitBatch({
  xCoordinates: data.x,
  yCoordinates: data.y,
  epochs: 100,
});

console.log("📊 Training Results:");
console.log(`   Final Loss: ${result.finalLoss.toFixed(6)}`);
console.log(`   Converged: ${result.converged}`);
console.log(`   Epochs: ${result.epochsCompleted}`);

// Make predictions
const predictions = model.predict(5);

console.log("\n🔮 Predictions:");
predictions.predictions.forEach((pred, i) => {
  console.log(
    `   Step ${i + 1}: [${pred.predicted.map((v) => v.toFixed(4)).join(", ")}]`,
  );
  console.log(
    `           ± [${pred.standardError.map((v) => v.toFixed(4)).join(", ")}]`,
  );
});
```

---

### Example 2: Online Learning with Drift Detection

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// Simulate streaming data with concept drift
async function streamingExample() {
  const model = new ConvolutionalRegression({
    hiddenLayers: 2,
    convolutionsPerLayer: 32,
    adwinDelta: 0.002,
    outlierThreshold: 3.0,
  });

  console.log("🌊 Starting streaming ingestion with drift detection...\n");

  let driftPoint = 500; // Drift occurs at sample 500

  for (let i = 0; i < 1000; i++) {
    // Generate data with concept drift
    let pattern: number;
    if (i < driftPoint) {
      // Initial pattern: linear
      pattern = i * 0.01 + Math.random() * 0.1;
    } else {
      // After drift: sinusoidal
      pattern = Math.sin(i * 0.1) + Math.random() * 0.1;
    }

    const x = [pattern, pattern * 2, pattern * 3];
    const y = [pattern * 4];

    const result = model.fitOnline({
      xCoordinates: [x],
      yCoordinates: [y],
    });

    // Log important events
    if (result.driftDetected) {
      console.log(`🚨 [Sample ${i}] DRIFT DETECTED!`);
      console.log(`   Loss: ${result.loss.toFixed(6)}`);
      console.log(`   Model adapting to new distribution...\n`);
    }

    if (result.isOutlier) {
      console.log(`📍 [Sample ${i}] Outlier detected (downweighted)`);
    }

    if (i % 100 === 0) {
      console.log(
        `📈 [Sample ${i}] Loss: ${result.loss.toFixed(6)}, LR: ${
          result.effectiveLearningRate.toExponential(2)
        }`,
      );
    }

    // Simulate streaming delay
    await new Promise((resolve) => setTimeout(resolve, 1));
  }

  const summary = model.getModelSummary();
  console.log("\n📊 Final Summary:");
  console.log(`   Samples processed: ${summary.sampleCount}`);
  console.log(`   Drift events: ${summary.driftCount}`);
  console.log(`   Accuracy: ${(summary.accuracy * 100).toFixed(2)}%`);
}

streamingExample();
```

---

### Example 3: Model Inspection and Debugging

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

function inspectModel() {
  const model = new ConvolutionalRegression({
    hiddenLayers: 2,
    convolutionsPerLayer: 16,
  });

  // Train with some data
  const trainData = {
    xCoordinates: Array.from(
      { length: 100 },
      () => Array.from({ length: 5 }, () => Math.random()),
    ),
    yCoordinates: Array.from(
      { length: 100 },
      () => Array.from({ length: 2 }, () => Math.random()),
    ),
  };

  model.fitBatch({ ...trainData, epochs: 50 });

  // Inspect model
  console.log("═══════════════════════════════════════════");
  console.log("          MODEL INSPECTION REPORT          ");
  console.log("═══════════════════════════════════════════\n");

  // 1. Model Summary
  const summary = model.getModelSummary();
  console.log("📋 MODEL SUMMARY");
  console.log("─────────────────");
  console.log(`   Input Dimension:  ${summary.inputDimension}`);
  console.log(`   Output Dimension: ${summary.outputDimension}`);
  console.log(`   Hidden Layers:    ${summary.hiddenLayers}`);
  console.log(`   Filters/Layer:    ${summary.convolutionsPerLayer}`);
  console.log(`   Kernel Size:      ${summary.kernelSize}`);
  console.log(
    `   Total Parameters: ${summary.totalParameters.toLocaleString()}`,
  );
  console.log(`   Current Accuracy: ${(summary.accuracy * 100).toFixed(2)}%`);
  console.log(`   Converged:        ${summary.converged}`);
  console.log();

  // 2. Normalization Stats
  const normStats = model.getNormalizationStats();
  console.log("📊 NORMALIZATION STATISTICS");
  console.log("───────────────────────────");
  console.log(`   Samples used: ${normStats.count}`);
  console.log(
    "   Input means:  ",
    normStats.inputMean.map((v) => v.toFixed(4)),
  );
  console.log("   Input stds:   ", normStats.inputStd.map((v) => v.toFixed(4)));
  console.log(
    "   Output means: ",
    normStats.outputMean.map((v) => v.toFixed(4)),
  );
  console.log(
    "   Output stds:  ",
    normStats.outputStd.map((v) => v.toFixed(4)),
  );
  console.log();

  // 3. Weight Statistics
  const weights = model.getWeights();
  console.log("⚖️  WEIGHT STATISTICS");
  console.log("────────────────────");
  console.log(`   Adam updates: ${weights.updateCount}`);

  weights.kernels.forEach((layer, i) => {
    const allWeights = layer.flat();
    const mean = allWeights.reduce((a, b) => a + b, 0) / allWeights.length;
    const std = Math.sqrt(
      allWeights.reduce((a, b) => a + (b - mean) ** 2, 0) / allWeights.length,
    );
    const max = Math.max(...allWeights.map(Math.abs));

    console.log(
      `   Layer ${i + 1}: mean=${mean.toFixed(6)}, std=${
        std.toFixed(6)
      }, max|w|=${max.toFixed(6)}`,
    );
  });

  console.log("\n═══════════════════════════════════════════\n");
}

inspectModel();
```

---

### Example 4: Hyperparameter Search

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

interface HyperparamConfig {
  hiddenLayers: number;
  convolutionsPerLayer: number;
  learningRate: number;
}

async function hyperparameterSearch(
  trainX: number[][],
  trainY: number[][],
  valX: number[][],
  valY: number[][],
) {
  const configs: HyperparamConfig[] = [
    { hiddenLayers: 1, convolutionsPerLayer: 16, learningRate: 0.01 },
    { hiddenLayers: 2, convolutionsPerLayer: 32, learningRate: 0.001 },
    { hiddenLayers: 2, convolutionsPerLayer: 64, learningRate: 0.001 },
    { hiddenLayers: 3, convolutionsPerLayer: 32, learningRate: 0.0005 },
    { hiddenLayers: 3, convolutionsPerLayer: 64, learningRate: 0.0005 },
  ];

  console.log("🔍 Hyperparameter Search");
  console.log("════════════════════════\n");

  const results: {
    config: HyperparamConfig;
    loss: number;
    accuracy: number;
  }[] = [];

  for (const config of configs) {
    const model = new ConvolutionalRegression(config);

    // Train
    model.fitBatch({
      xCoordinates: trainX,
      yCoordinates: trainY,
      epochs: 50,
    });

    // Validate
    let valLoss = 0;
    for (let i = 0; i < valX.length; i++) {
      const result = model.fitOnline({
        xCoordinates: [valX[i]],
        yCoordinates: [valY[i]],
      });
      valLoss += result.loss;
    }
    valLoss /= valX.length;

    const summary = model.getModelSummary();

    results.push({
      config,
      loss: valLoss,
      accuracy: summary.accuracy,
    });

    console.log(
      `Config: L=${config.hiddenLayers}, C=${config.convolutionsPerLayer}, LR=${config.learningRate}`,
    );
    console.log(
      `  → Val Loss: ${valLoss.toFixed(6)}, Accuracy: ${
        (summary.accuracy * 100).toFixed(2)
      }%\n`,
    );
  }

  // Find best
  results.sort((a, b) => a.loss - b.loss);

  console.log("🏆 Best Configuration:");
  console.log(`   Hidden Layers: ${results[0].config.hiddenLayers}`);
  console.log(`   Filters/Layer: ${results[0].config.convolutionsPerLayer}`);
  console.log(`   Learning Rate: ${results[0].config.learningRate}`);
  console.log(`   Validation Loss: ${results[0].loss.toFixed(6)}`);
}
```

---

## 🔬 Algorithms & Concepts

### Welford's Online Algorithm

Used for numerically stable computation of running mean and variance.

```
For each new sample xₙ:
    δ = xₙ - μₙ₋₁
    μₙ = μₙ₋₁ + δ/n
    M₂,ₙ = M₂,ₙ₋₁ + δ × (xₙ - μₙ)
    
Variance: σ² = M₂/(n-1)
```

**Why it matters:** Standard variance calculation (Σ(x-μ)²) can suffer from
catastrophic cancellation. Welford's algorithm maintains numerical stability
even after millions of samples.

---

### Adam Optimizer

Adaptive Moment Estimation optimizer with bias correction.

```
For each parameter w with gradient g:
    m = β₁·m + (1-β₁)·g          # First moment (momentum)
    v = β₂·v + (1-β₂)·g²         # Second moment (RMSprop)
    
    m̂ = m / (1-β₁ᵗ)              # Bias correction
    v̂ = v / (1-β₂ᵗ)              # Bias correction
    
    w = w - η·m̂/(√v̂ + ε)        # Update
```

**Benefits:**

- Combines momentum and adaptive learning rates
- Works well with sparse gradients
- Requires little tuning

---

### ADWIN Drift Detection

ADaptive WINdowing algorithm for detecting distribution changes.

```
Sliding Window:
    ┌─────────────────────────────────────────┐
    │  W₀ (older)  │  W₁ (newer)             │
    └─────────────────────────────────────────┘
    
Cut threshold:
    εcut = √((1/2m)·ln(4/δ))
    
Drift detected when:
    |μ(W₀) - μ(W₁)| ≥ εcut
```

**When drift is detected:**

1. Older window is discarded
2. Normalization statistics are decayed
3. Model adapts to new distribution

---

### He Initialization

Weight initialization optimized for ReLU activations.

```
W ~ N(0, √(2/fan_in))

Where fan_in = input_channels × kernel_size
```

**Why it matters:** Proper initialization prevents vanishing/exploding gradients
and enables training of deep networks.

---

## ⚡ Performance Optimization

### Memory Efficiency

```typescript
// The library uses preallocated Float64Arrays
// No garbage collection during forward/backward passes

// ✅ Good: Reuse model for streaming
const model = new ConvolutionalRegression();
for (const sample of stream) {
  model.fitOnline(sample); // No allocations!
}

// ❌ Bad: Creating new models
for (const sample of stream) {
  const model = new ConvolutionalRegression(); // GC pressure!
  model.fitOnline(sample);
}
```

### Batch Size Optimization

```typescript
// Larger batches = better GPU/SIMD utilization
// but watch memory usage

// For CPU: 32-64 usually optimal
const cpuModel = new ConvolutionalRegression({
  batchSize: 32,
});

// For GPU (if using WebGL/WebGPU backend): 128-256
const gpuModel = new ConvolutionalRegression({
  batchSize: 256,
});
```

### Architecture Sizing

```
Total Parameters ≈ 
    hiddenLayers × convolutionsPerLayer² × kernelSize +
    convolutionsPerLayer × inputDim × outputDim

Example (default config with inputDim=10, outputDim=2):
    2 × 32² × 3 + 32 × 10 × 2 = 6,784 parameters
```

---

## 🔧 Troubleshooting

### Common Issues

#### Loss Not Decreasing

```typescript
// Problem: Learning rate too low or too high

// Solution 1: Increase learning rate
const model = new ConvolutionalRegression({
  learningRate: 0.01, // Try 10x higher
});

// Solution 2: Check data normalization
const stats = model.getNormalizationStats();
console.log("Input std:", stats.inputStd); // Should be ~1 after normalization

// Solution 3: Reduce regularization
const model = new ConvolutionalRegression({
  regularizationStrength: 1e-5, // Reduce if underfitting
});
```

#### Loss Exploding (NaN/Infinity)

```typescript
// Problem: Numerical instability

// Solution 1: Lower learning rate
const model = new ConvolutionalRegression({
  learningRate: 0.0001,
  warmupSteps: 500, // Longer warmup
});

// Solution 2: Check for outliers in data
// The model handles outliers, but extreme values can cause issues

// Solution 3: Increase epsilon
const model = new ConvolutionalRegression({
  epsilon: 1e-6, // More numerical stability
});
```

#### Poor Generalization

```typescript
// Problem: Overfitting

// Solution 1: Increase regularization
const model = new ConvolutionalRegression({
  regularizationStrength: 1e-3,
});

// Solution 2: Reduce model capacity
const model = new ConvolutionalRegression({
  hiddenLayers: 1,
  convolutionsPerLayer: 16,
});

// Solution 3: Early stopping
const result = model.fitBatch({
  xCoordinates: trainX,
  yCoordinates: trainY,
  epochs: 1000, // Will stop early if converged
});
```

#### Too Many False Drift Detections

```typescript
// Problem: ADWIN is too sensitive

// Solution: Increase delta (reduce sensitivity)
const model = new ConvolutionalRegression({
  adwinDelta: 0.01, // Less sensitive to changes
});
```

---

## 📄 License

MIT License - feel free to use in personal and commercial projects.

---

<div align="center">

**Built with ❤️ for high-performance machine learning in TypeScript**

[Report Bug](https://github.com/@hviana/multivariate-convolutional-regression/issues)
•
[Request Feature](https://github.com/@hviana/multivariate-convolutional-regression/issues)
•
[Contribute](https://github.com/@hviana/multivariate-convolutional-regression/pulls)

</div>
