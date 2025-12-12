Model: # 🧠 ConvolutionalRegression

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     ██████╗ ██████╗ ███╗   ██╗██╗   ██╗ ██████╗ ██╗     ██╗   ██╗████████╗║
║    ██╔════╝██╔═══██╗████╗  ██║██║   ██║██╔═══██╗██║     ██║   ██║╚══██╔══╝║
║    ██║     ██║   ██║██╔██╗ ██║██║   ██║██║   ██║██║     ██║   ██║   ██║   ║
║    ██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝██║   ██║██║     ██║   ██║   ██║   ║
║    ╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ ╚██████╔╝███████╗╚██████╔╝   ██║   ║
║     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝   ╚═════╝ ╚══════╝ ╚═════╝    ╚═╝   ║
║                                                                           ║
║             🔮 REGRESSION • 🌊 ONLINE LEARNING • ⚡ ADAM                   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**A powerful 1D Convolutional Neural Network for Multivariate Regression with
Incremental Online Learning**

[Features](#-features) • [Quick Start](#-quick-start) • [API](#-api-reference) •
[Examples](#-examples) • [Parameters](#-parameter-optimization-guide)

</div>

---

## 📑 Table of Contents

<details>
<summary>Click to expand</summary>

- [🌟 Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [📚 API Reference](#-api-reference)
  - [Configuration](#configuration)
  - [Methods](#methods)
  - [Interfaces](#interfaces)
- [⚙️ Parameter Optimization Guide](#️-parameter-optimization-guide)
- [📊 Examples](#-examples)
- [🔬 Mathematical Background](#-mathematical-background)
- [🎯 Use Cases](#-use-cases)
- [❓ FAQ](#-faq)
- [🤝 Contributing](#-contributing)

</details>

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### 🧬 Core Capabilities

| Feature                 | Description                           |
| ----------------------- | ------------------------------------- |
| 🔄 **Online Learning**  | Train incrementally on streaming data |
| 🎯 **Multivariate I/O** | Handle multiple inputs and outputs    |
| 📈 **Auto-scaling**     | Automatic Z-score normalization       |
| 🛡️ **Drift Detection**  | ADWIN algorithm for concept drift     |

</td>
<td width="50%">

### ⚡ Optimizations

| Feature                  | Description                     |
| ------------------------ | ------------------------------- |
| 🚀 **Adam Optimizer**    | Adaptive learning with momentum |
| 📉 **LR Scheduling**     | Warmup + Cosine decay           |
| 🔒 **L2 Regularization** | Built-in overfitting prevention |
| 🎲 **Outlier Detection** | Z-score based anomaly filtering |

</td>
</tr>
</table>

### 🎨 Feature Highlights

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ✅ Zero Dependencies      ✅ TypeScript Native    ✅ Memory Efficient │
│   ✅ Confidence Intervals   ✅ Save/Load State      ✅ Auto Initialization│
│   ✅ He Weight Init         ✅ Welford's Algorithm  ✅ Buffer Pooling    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Basic Usage

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// 1️⃣ Create model instance
const model = new ConvolutionalRegression({
  hiddenLayers: 2,
  convolutionsPerLayer: 32,
  learningRate: 0.001,
});

// 2️⃣ Train incrementally with data
const result = model.fitOnline({
  xCoordinates: [[1, 2, 3, 4, 5]],
  yCoordinates: [[6, 7]],
});

console.log(`📊 Loss: ${result.loss.toFixed(4)}`);
console.log(`📈 Converged: ${result.converged}`);

// 3️⃣ Make predictions
const predictions = model.predict(3);

predictions.predictions.forEach((pred, i) => {
  console.log(
    `🔮 Step ${i + 1}: ${pred.predicted.map((v) => v.toFixed(2)).join(", ")}`,
  );
  console.log(
    `   95% CI: [${pred.lowerBound.map((v) => v.toFixed(2)).join(", ")}] - [${
      pred.upperBound.map((v) => v.toFixed(2)).join(", ")
    }]`,
  );
});
```

### 📋 Output Example

```
📊 Loss: 0.4523
📈 Converged: false
🔮 Step 1: 6.12, 7.08
   95% CI: [5.82, 6.78] - [6.42, 7.38]
🔮 Step 2: 6.25, 7.15
   95% CI: [5.89, 6.75] - [6.61, 7.55]
🔮 Step 3: 6.31, 7.22
   95% CI: [5.88, 6.68] - [6.74, 7.76]
```

---

## 🏗️ Architecture

### Network Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CONVOLUTIONAL REGRESSION NETWORK                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────┐    ┌─────────────────┐         ┌─────────────────┐   ┌─────────┐ │
│   │  INPUT  │    │   CONV BLOCK 1  │   ...   │   CONV BLOCK N  │   │ OUTPUT  │ │
│   │         │    │                 │         │                 │   │         │ │
│   │ (d_in)  │───▶│  Conv1D + ReLU  │────────▶│  Conv1D + ReLU  │──▶│ DENSE   │ │
│   │         │    │                 │         │                 │   │         │ │
│   │ [1×d]   │    │  [f×d] filters  │         │  [f×d] filters  │   │ (d_out) │ │
│   └─────────┘    └─────────────────┘         └─────────────────┘   └─────────┘ │
│                                                                                  │
│   Legend: d_in = input dim, d = spatial dim, f = filters, d_out = output dim    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
                          TRAINING PIPELINE
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   ┌────────┐   ┌────────────┐   ┌──────────┐   ┌───────────────┐  │
│   │  Raw   │   │  Welford   │   │   Z-Score │   │   Forward    │  │
│   │  Data  │──▶│  Update    │──▶│ Normalize │──▶│    Pass      │  │
│   └────────┘   └────────────┘   └──────────┘   └───────┬───────┘  │
│                                                        │          │
│                                                        ▼          │
│   ┌────────┐   ┌────────────┐   ┌──────────┐   ┌───────────────┐  │
│   │ Update │   │    Adam    │   │ Backward │   │   Compute    │  │
│   │Weights │◀──│  Optimizer │◀──│   Pass   │◀──│    Loss      │  │
│   └────────┘   └────────────┘   └──────────┘   └───────────────┘  │
│        │                                                          │
│        ▼                                                          │
│   ┌────────────────────────────────────────────────────────┐     │
│   │               ADWIN Drift Detection                    │     │
│   │         (Monitor loss distribution changes)            │     │
│   └────────────────────────────────────────────────────────┘     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Convolution Operation

```
                    1D CONVOLUTION WITH SAME PADDING

  Input (1 channel)          Kernel (k=3)           Output (1 filter)
┌─────────────────┐        ┌───────────┐         ┌─────────────────┐
│ x₀ x₁ x₂ x₃ x₄ │   *    │ w₀ w₁ w₂ │    =    │ y₀ y₁ y₂ y₃ y₄ │
└─────────────────┘        └───────────┘         └─────────────────┘

Formula: y[i] = Σₖ (w[k] · x[i+k-pad]) + bias

Padding: pad = ⌊kernel_size / 2⌋ (same padding preserves spatial dims)
```

---

## 📚 API Reference

### Configuration

#### `ConvolutionalRegressionConfig`

Create a model with custom configuration:

```typescript
const model = new ConvolutionalRegression({
  // Architecture
  hiddenLayers: 2, // Number of conv layers (1-10)
  convolutionsPerLayer: 32, // Filters per layer (1-256)
  kernelSize: 3, // Convolution kernel size

  // Learning Rate
  learningRate: 0.001, // Base learning rate
  warmupSteps: 100, // Linear warmup steps
  totalSteps: 10000, // Total steps for cosine decay

  // Adam Optimizer
  beta1: 0.9, // First moment decay
  beta2: 0.999, // Second moment decay
  epsilon: 1e-8, // Numerical stability

  // Regularization
  regularizationStrength: 1e-4, // L2 regularization λ

  // Convergence
  convergenceThreshold: 1e-6, // Loss threshold

  // Outlier Detection
  outlierThreshold: 3.0, // Z-score threshold

  // Drift Detection
  adwinDelta: 0.002, // ADWIN confidence parameter
});
```

#### Configuration Parameters Table

| Parameter                | Type     | Default | Range   | Description                     |
| ------------------------ | -------- | ------- | ------- | ------------------------------- |
| `hiddenLayers`           | `number` | `2`     | `1-10`  | Number of convolutional layers  |
| `convolutionsPerLayer`   | `number` | `32`    | `1-256` | Filters per convolutional layer |
| `kernelSize`             | `number` | `3`     | `≥1`    | Size of convolution kernel      |
| `learningRate`           | `number` | `0.001` | `>0`    | Base learning rate for Adam     |
| `warmupSteps`            | `number` | `100`   | `≥0`    | Steps for linear LR warmup      |
| `totalSteps`             | `number` | `10000` | `≥1`    | Total steps for cosine decay    |
| `beta1`                  | `number` | `0.9`   | `[0,1)` | Adam first moment decay         |
| `beta2`                  | `number` | `0.999` | `[0,1)` | Adam second moment decay        |
| `epsilon`                | `number` | `1e-8`  | `>0`    | Adam numerical stability        |
| `regularizationStrength` | `number` | `1e-4`  | `≥0`    | L2 regularization strength      |
| `convergenceThreshold`   | `number` | `1e-6`  | `>0`    | Loss convergence threshold      |
| `outlierThreshold`       | `number` | `3.0`   | `>0`    | Z-score outlier threshold       |
| `adwinDelta`             | `number` | `0.002` | `(0,1)` | ADWIN confidence delta          |

---

### Methods

#### 🎓 `fitOnline(data)`

Performs incremental online training with a single batch of data.

```typescript
fitOnline(data: {
  xCoordinates: number[][],
  yCoordinates: number[][]
}): FitResult
```

**Parameters:**

| Name                | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| `data.xCoordinates` | `number[][]` | Input features `[batch][features]` |
| `data.yCoordinates` | `number[][]` | Target outputs `[batch][outputs]`  |

**Returns:** `FitResult`

```typescript
interface FitResult {
  loss: number; // Current loss value
  gradientNorm: number; // L2 norm of gradients
  effectiveLearningRate: number; // Current learning rate
  isOutlier: boolean; // Outlier detection flag
  converged: boolean; // Convergence status
  sampleIndex: number; // Current sample count
  driftDetected: boolean; // Drift detection flag
}
```

**Example:**

```typescript
// Single sample training
const result = model.fitOnline({
  xCoordinates: [[1.0, 2.0, 3.0, 4.0, 5.0]],
  yCoordinates: [[10.0, 12.0]],
});

// Batch training
const batchResult = model.fitOnline({
  xCoordinates: [
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [2.0, 3.0, 4.0, 5.0, 6.0],
    [3.0, 4.0, 5.0, 6.0, 7.0],
  ],
  yCoordinates: [
    [10.0, 12.0],
    [12.0, 14.0],
    [14.0, 16.0],
  ],
});

console.log(`Loss: ${result.loss}`);
console.log(`Gradient Norm: ${result.gradientNorm}`);
console.log(`Learning Rate: ${result.effectiveLearningRate}`);
```

---

#### 🔮 `predict(futureSteps)`

Generates predictions for future steps using autoregressive prediction.

```typescript
predict(futureSteps: number): PredictionResult
```

**Parameters:**

| Name          | Type     | Description                       |
| ------------- | -------- | --------------------------------- |
| `futureSteps` | `number` | Number of future steps to predict |

**Returns:** `PredictionResult`

```typescript
interface PredictionResult {
  predictions: SinglePrediction[]; // Array of predictions
  accuracy: number; // Model accuracy: 1/(1 + L̄)
  sampleCount: number; // Training sample count
  isModelReady: boolean; // Model readiness flag
}

interface SinglePrediction {
  predicted: number[]; // Predicted values
  lowerBound: number[]; // 95% CI lower bound
  upperBound: number[]; // 95% CI upper bound
  standardError: number[]; // Standard error per dimension
}
```

**Example:**

```typescript
const predictions = model.predict(5);

if (predictions.isModelReady) {
  predictions.predictions.forEach((pred, step) => {
    console.log(`\n📅 Step ${step + 1}:`);
    console.log(`   Predicted: [${pred.predicted.join(", ")}]`);
    console.log(`   Lower 95%: [${pred.lowerBound.join(", ")}]`);
    console.log(`   Upper 95%: [${pred.upperBound.join(", ")}]`);
    console.log(`   Std Error: [${pred.standardError.join(", ")}]`);
  });

  console.log(
    `\n📊 Model Accuracy: ${(predictions.accuracy * 100).toFixed(2)}%`,
  );
}
```

---

#### 📊 `getModelSummary()`

Returns comprehensive model information.

```typescript
getModelSummary(): ModelSummary
```

**Returns:** `ModelSummary`

```typescript
interface ModelSummary {
  isInitialized: boolean; // Initialization status
  inputDimension: number; // Input feature count
  outputDimension: number; // Output feature count
  hiddenLayers: number; // Number of conv layers
  convolutionsPerLayer: number; // Filters per layer
  kernelSize: number; // Convolution kernel size
  totalParameters: number; // Total trainable params
  sampleCount: number; // Processed samples
  accuracy: number; // Current accuracy
  converged: boolean; // Convergence status
  effectiveLearningRate: number; // Current learning rate
  driftCount: number; // Detected drift events
}
```

**Example:**

```typescript
const summary = model.getModelSummary();

console.log(`
╔════════════════════════════════════════╗
║           MODEL SUMMARY                ║
╠════════════════════════════════════════╣
║ Status:      ${summary.isInitialized ? "✅ Initialized" : "❌ Not Ready"}
║ Architecture: ${summary.inputDimension} → [Conv×${summary.hiddenLayers}] → ${summary.outputDimension}
║ Parameters:  ${summary.totalParameters.toLocaleString()}
║ Samples:     ${summary.sampleCount.toLocaleString()}
║ Accuracy:    ${(summary.accuracy * 100).toFixed(2)}%
║ Converged:   ${summary.converged ? "✅ Yes" : "⏳ No"}
║ Drift Events: ${summary.driftCount}
╚════════════════════════════════════════╝
`);
```

---

#### 🔧 `getWeights()`

Retrieves model weights and optimizer states.

```typescript
getWeights(): WeightInfo
```

**Returns:** `WeightInfo`

```typescript
interface WeightInfo {
  kernels: number[][][]; // Conv kernel weights
  biases: number[][]; // Bias terms
  firstMoment: number[][][]; // Adam m values
  secondMoment: number[][][]; // Adam v values
  updateCount: number; // Total updates
}
```

---

#### 📈 `getNormalizationStats()`

Gets running normalization statistics.

```typescript
getNormalizationStats(): NormalizationStats
```

**Returns:** `NormalizationStats`

```typescript
interface NormalizationStats {
  inputMean: number[]; // Running input mean
  inputStd: number[]; // Running input std
  outputMean: number[]; // Running output mean
  outputStd: number[]; // Running output std
  count: number; // Sample count
}
```

---

#### 🔄 `reset()`

Resets model to initial state.

```typescript
reset(): void
```

**Example:**

```typescript
model.reset();
console.log("Model reset to initial state");
```

---

#### 💾 `save()`

Serializes model state to JSON string.

```typescript
save(): string
```

**Example:**

```typescript
// Save to localStorage
const modelState = model.save();
localStorage.setItem("my-model", modelState);

// Save to file (Node.js)
import { writeFileSync } from "fs";
writeFileSync("model.json", model.save());
```

---

#### 📂 `load(jsonString)`

Loads model state from JSON string.

```typescript
load(jsonString: string): void
```

**Example:**

```typescript
// Load from localStorage
const savedState = localStorage.getItem("my-model");
if (savedState) {
  model.load(savedState);
  console.log("Model loaded successfully!");
}

// Load from file (Node.js)
import { readFileSync } from "fs";
model.load(readFileSync("model.json", "utf-8"));
```

---

## ⚙️ Parameter Optimization Guide

### 🎯 Quick Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER SELECTION DECISION TREE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         What's your priority?                               │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼               ▼                              │
│         ⚡ Speed        🎯 Accuracy      💾 Memory                          │
│              │               │               │                              │
│              ▼               ▼               ▼                              │
│     hiddenLayers: 1    hiddenLayers: 3+   hiddenLayers: 1-2                │
│     filters: 16        filters: 64-128    filters: 16-32                   │
│     kernelSize: 3      kernelSize: 5-7    kernelSize: 3                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 📊 Parameter Deep Dive

#### 1️⃣ Hidden Layers (`hiddenLayers`)

Controls network depth and feature hierarchy.

```
DEPTH VS COMPLEXITY

Shallow (1-2 layers)          Deep (3-5 layers)           Very Deep (6-10 layers)
━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━
✅ Fast training              ✅ Complex patterns          ⚠️ Risk of overfitting
✅ Less overfitting           ✅ Better abstraction        ⚠️ Slow convergence
✅ Good for simple data       ⚠️ More data needed          ⚠️ Requires more data
❌ Limited expressivity       ❌ Slower training           ✅ Maximum expressivity
```

| Use Case            | Recommended | Example                               |
| ------------------- | ----------- | ------------------------------------- |
| Simple regression   | `1-2`       | Linear trends, simple patterns        |
| Standard timeseries | `2-3`       | Stock prices, weather data            |
| Complex patterns    | `3-5`       | Multi-seasonal data, complex dynamics |
| High-dimensional    | `4-6`       | Image-like sequential data            |

**Example Configuration:**

```typescript
// Simple linear trends
const simpleModel = new ConvolutionalRegression({
  hiddenLayers: 1,
  convolutionsPerLayer: 16,
});

// Complex seasonal patterns
const complexModel = new ConvolutionalRegression({
  hiddenLayers: 4,
  convolutionsPerLayer: 64,
});
```

---

#### 2️⃣ Convolutions Per Layer (`convolutionsPerLayer`)

Controls the number of learned features at each layer.

```
FILTER COUNT IMPACT

      Low (8-16)              Medium (32-64)            High (128-256)
    ┌──────────┐            ┌──────────────┐          ┌──────────────┐
    │ ░░░░░░░░ │            │ ▓▓▓▓▓▓▓▓▓▓▓▓ │          │ ████████████ │
    │ ░░░░░░░░ │            │ ▓▓▓▓▓▓▓▓▓▓▓▓ │          │ ████████████ │
    └──────────┘            └──────────────┘          └──────────────┘
    ~100-1K params          ~10K-50K params          ~100K-500K params
    
    ✅ Fast inference        ✅ Balanced                ✅ High capacity
    ❌ Limited features      ✅ Good generalization     ❌ Memory intensive
```

**Scaling Formula:**

```
Parameters ≈ Σ(filters[l] × filters[l-1] × kernel_size) + dense_layer
```

| Data Complexity | Recommended Filters | Total Params (approx) |
| --------------- | ------------------- | --------------------- |
| Low             | `8-16`              | ~500 - 2K             |
| Medium          | `32-64`             | ~10K - 50K            |
| High            | `64-128`            | ~50K - 200K           |
| Very High       | `128-256`           | ~200K - 1M            |

**Example:**

```typescript
// Resource-constrained environment
const lightModel = new ConvolutionalRegression({
  hiddenLayers: 2,
  convolutionsPerLayer: 16, // ~2K parameters
});

// High-accuracy requirement
const heavyModel = new ConvolutionalRegression({
  hiddenLayers: 3,
  convolutionsPerLayer: 128, // ~150K parameters
});
```

---

#### 3️⃣ Kernel Size (`kernelSize`)

Controls the receptive field of each convolution.

```
RECEPTIVE FIELD VISUALIZATION

kernelSize = 3                kernelSize = 5                kernelSize = 7
───●───                       ─────●─────                   ───────●───────
   ↓                              ↓                              ↓
[x][x][x]                  [x][x][x][x][x]            [x][x][x][x][x][x][x]
   │                              │                              │
Local patterns            Medium-range patterns       Long-range patterns
```

| Pattern Type      | Kernel Size           | Use Case                           |
| ----------------- | --------------------- | ---------------------------------- |
| Sharp changes     | `3`                   | Point anomalies, quick transitions |
| Smooth trends     | `5-7`                 | Gradual changes, moving averages   |
| Long dependencies | `7-11`                | Seasonal patterns, slow dynamics   |
| Mixed             | `3` + multiple layers | Hierarchical feature extraction    |

**Example:**

```typescript
// Quick-changing signals (e.g., high-frequency trading)
const quickModel = new ConvolutionalRegression({
  kernelSize: 3,
  hiddenLayers: 3, // Stack to increase receptive field
});

// Slow-changing signals (e.g., climate data)
const slowModel = new ConvolutionalRegression({
  kernelSize: 7,
  hiddenLayers: 2,
});
```

---

#### 4️⃣ Learning Rate (`learningRate`)

Controls the step size during optimization.

```
LEARNING RATE EFFECTS

Too High (>0.01)          Optimal (~0.001)           Too Low (<0.0001)
        ╱╲                       ╱╲                        
       ╱  ╲                     ╱  ╲                      ___
      ╱    ╲                   ╱    ╲___                 ╱
     ╱      ╲                 ╱         ───             ╱
    ╱        ╲               ╱                         ╱
───╱──────────╲───       ───╱──────────────────    ───╱────────────────
   Divergence!            Smooth convergence        Very slow progress
```

**Recommended Values:**

| Scenario      | Learning Rate | Notes               |
| ------------- | ------------- | ------------------- |
| Default       | `0.001`       | Good starting point |
| Fine-tuning   | `0.0001`      | Small adjustments   |
| Noisy data    | `0.0005`      | More stable         |
| Large batches | `0.003`       | Can scale up        |
| Small batches | `0.0005`      | Reduce variance     |

---

#### 5️⃣ Learning Rate Schedule (`warmupSteps`, `totalSteps`)

```
LEARNING RATE SCHEDULE

     LR
      │
  max │     ╱╲
      │    ╱  ╲
      │   ╱    ╲____
      │  ╱          ╲____
      │ ╱                ╲____
  min │╱                      ╲___
      └────────────────────────────► Steps
        ↑           ↑
      Warmup    Cosine Decay
      Phase       Phase

Formula:
  Warmup:  lr = base_lr × (step / warmup_steps)
  Decay:   lr = base_lr × 0.5 × (1 + cos(π × progress))
```

| Dataset Size       | Warmup Steps | Total Steps |
| ------------------ | ------------ | ----------- |
| Small (<1K)        | `50`         | `2000`      |
| Medium (1K-10K)    | `100`        | `10000`     |
| Large (10K-100K)   | `500`        | `50000`     |
| Very Large (>100K) | `1000`       | `100000`    |

**Example:**

```typescript
// Small dataset with quick training
const smallDataModel = new ConvolutionalRegression({
  warmupSteps: 50,
  totalSteps: 2000,
  learningRate: 0.001,
});

// Large dataset with extended training
const largeDataModel = new ConvolutionalRegression({
  warmupSteps: 1000,
  totalSteps: 100000,
  learningRate: 0.001,
});
```

---

#### 6️⃣ Adam Optimizer (`beta1`, `beta2`, `epsilon`)

```
ADAM MOMENTUM VISUALIZATION

                    Gradient with noise
                    ↓  ↓  ↓  ↓  ↓  ↓
                    ●──●──●──●──●──●   Raw gradients (noisy)
                   
                         β₁ = 0.9
                    ↓    (momentum)    ↓
                    ●═══════════════●   First moment (smoothed direction)
                    
                         β₂ = 0.999
                    ↓   (RMSprop)     ↓
                    ●═══════════════●   Second moment (adaptive LR)
```

| Parameter | Default | Effect              | Tuning                          |
| --------- | ------- | ------------------- | ------------------------------- |
| `beta1`   | `0.9`   | Momentum decay      | Lower (0.8) for noisy gradients |
| `beta2`   | `0.999` | Adaptive LR decay   | Lower (0.99) for non-stationary |
| `epsilon` | `1e-8`  | Numerical stability | Increase for sparse gradients   |

---

#### 7️⃣ Regularization (`regularizationStrength`)

```
L2 REGULARIZATION EFFECT

     Loss
      │
      │  ●────────●  No regularization (overfitting)
      │   ╲
      │    ●──────●  Moderate λ (good fit)
      │     ╲
      │      ●────●  High λ (underfitting)
      │
      └────────────────► Model Complexity

Loss = MSE + (λ/2) × Σ‖W‖²
```

| Scenario                    | λ Value | Effect                |
| --------------------------- | ------- | --------------------- |
| Complex model, limited data | `1e-3`  | Strong regularization |
| Balanced                    | `1e-4`  | Default, moderate     |
| Simple model, lots of data  | `1e-5`  | Light regularization  |
| No regularization           | `0`     | Pure MSE loss         |

---

#### 8️⃣ Outlier Detection (`outlierThreshold`)

```
OUTLIER DETECTION

     │  Normal Distribution
     │        ╱╲
     │       ╱  ╲
     │      ╱    ╲
     │     ╱      ╲
     │────╱────────╲────
         │←──3σ──→│
         
    Points outside 3σ are flagged as outliers
    and given reduced weight (0.1× instead of 1×)
```

| Threshold | Outlier % (Normal) | Use Case                |
| --------- | ------------------ | ----------------------- |
| `2.0`     | ~4.6%              | Aggressive filtering    |
| `2.5`     | ~1.2%              | Moderate filtering      |
| `3.0`     | ~0.3%              | Conservative (default)  |
| `4.0`     | ~0.006%            | Very rare outliers only |

---

#### 9️⃣ Drift Detection (`adwinDelta`)

```
ADWIN DRIFT DETECTION

     Loss
      │    Concept Drift!
      │         ↓
      │  ●●●●●●●╱●●●●●●●
      │        ╱
      │  ●●●●●╱
      │      │
      └──────┴────────────► Time
            Window shrinks when
            drift is detected
```

| Delta    | Sensitivity      | Use Case                |
| -------- | ---------------- | ----------------------- |
| `0.0002` | Very High        | Rapid adaptation needed |
| `0.002`  | Medium (default) | Balanced detection      |
| `0.02`   | Low              | Only major shifts       |
| `0.2`    | Very Low         | Ignore most changes     |

---

### 🎛️ Preset Configurations

```typescript
// 🚀 FAST: Minimal computation, quick results
const FAST_PRESET = {
  hiddenLayers: 1,
  convolutionsPerLayer: 16,
  kernelSize: 3,
  learningRate: 0.003,
  warmupSteps: 50,
  totalSteps: 2000,
};

// ⚖️ BALANCED: Good accuracy with reasonable speed
const BALANCED_PRESET = {
  hiddenLayers: 2,
  convolutionsPerLayer: 32,
  kernelSize: 3,
  learningRate: 0.001,
  warmupSteps: 100,
  totalSteps: 10000,
};

// 🎯 ACCURATE: Maximum accuracy, slower training
const ACCURATE_PRESET = {
  hiddenLayers: 4,
  convolutionsPerLayer: 64,
  kernelSize: 5,
  learningRate: 0.0005,
  warmupSteps: 500,
  totalSteps: 50000,
  regularizationStrength: 1e-5,
};

// 🌊 STREAMING: Optimized for online learning
const STREAMING_PRESET = {
  hiddenLayers: 2,
  convolutionsPerLayer: 32,
  kernelSize: 3,
  learningRate: 0.001,
  adwinDelta: 0.001, // More sensitive drift detection
  outlierThreshold: 2.5, // More aggressive outlier filtering
};
```

---

## 📊 Examples

### Example 1: Time Series Forecasting

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// Create model for stock price prediction
const stockModel = new ConvolutionalRegression({
  hiddenLayers: 3,
  convolutionsPerLayer: 64,
  kernelSize: 5,
  learningRate: 0.001,
  outlierThreshold: 2.5, // Financial data often has outliers
});

// Training data: sliding window of 10 days → predict next 2 days
const trainingData = {
  xCoordinates: [
    [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
    [102, 101, 103, 105, 104, 106, 108, 107, 109, 111],
    [101, 103, 105, 104, 106, 108, 107, 109, 111, 110],
  ],
  yCoordinates: [
    [111, 110],
    [110, 112],
    [112, 114],
  ],
};

// Train for multiple epochs
for (let epoch = 0; epoch < 100; epoch++) {
  const result = stockModel.fitOnline(trainingData);

  if (epoch % 20 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${result.loss.toFixed(6)}`);
  }

  if (result.converged) {
    console.log(`✅ Converged at epoch ${epoch}`);
    break;
  }
}

// Predict next 5 days
const forecast = stockModel.predict(5);
console.log("\n📈 Stock Price Forecast:");
forecast.predictions.forEach((pred, day) => {
  console.log(
    `  Day ${day + 1}: $${pred.predicted[0].toFixed(2)} (±${
      (pred.upperBound[0] - pred.predicted[0]).toFixed(2)
    })`,
  );
});
```

---

### Example 2: Multi-Sensor Prediction

```typescript
// IoT sensor data: temperature, humidity, pressure → predict future values
const sensorModel = new ConvolutionalRegression({
  hiddenLayers: 2,
  convolutionsPerLayer: 48,
  learningRate: 0.001,
});

// Simulate streaming sensor data
function processSensorReading(
  temp: number,
  humidity: number,
  pressure: number,
) {
  // Historical window (last 6 readings)
  const inputWindow = [
    prevReadings.slice(-6).map((r) => r.temp),
    prevReadings.slice(-6).map((r) => r.humidity),
    prevReadings.slice(-6).map((r) => r.pressure),
  ].flat();

  const result = sensorModel.fitOnline({
    xCoordinates: [inputWindow],
    yCoordinates: [[temp, humidity, pressure]],
  });

  if (result.driftDetected) {
    console.log("⚠️ Environmental change detected!");
  }

  return result;
}

// Get predictions with confidence intervals
const predictions = sensorModel.predict(3);
predictions.predictions.forEach((pred, hour) => {
  console.log(`\nHour +${hour + 1}:`);
  console.log(
    `  🌡️  Temp: ${pred.predicted[0].toFixed(1)}°C [${
      pred.lowerBound[0].toFixed(1)
    } - ${pred.upperBound[0].toFixed(1)}]`,
  );
  console.log(
    `  💧 Humidity: ${pred.predicted[1].toFixed(1)}% [${
      pred.lowerBound[1].toFixed(1)
    } - ${pred.upperBound[1].toFixed(1)}]`,
  );
  console.log(
    `  📊 Pressure: ${pred.predicted[2].toFixed(1)}hPa [${
      pred.lowerBound[2].toFixed(1)
    } - ${pred.upperBound[2].toFixed(1)}]`,
  );
});
```

---

### Example 3: Model Persistence

```typescript
// Save trained model
const model = new ConvolutionalRegression({ hiddenLayers: 3 });

// ... training ...

// Save to JSON
const savedState = model.save();
console.log(`Model size: ${(savedState.length / 1024).toFixed(2)} KB`);

// Store in localStorage
localStorage.setItem("myModel", savedState);

// Later: Load model
const loadedModel = new ConvolutionalRegression();
loadedModel.load(localStorage.getItem("myModel")!);

// Continue training or make predictions
const summary = loadedModel.getModelSummary();
console.log(`Loaded model with ${summary.sampleCount} training samples`);
```

---

### Example 4: Real-time Dashboard Integration

```typescript
class PredictionDashboard {
  private model: ConvolutionalRegression;
  private dataBuffer: number[][] = [];

  constructor() {
    this.model = new ConvolutionalRegression({
      hiddenLayers: 2,
      convolutionsPerLayer: 32,
      adwinDelta: 0.001, // Quick drift detection
    });
  }

  addDataPoint(values: number[]): {
    prediction: number[] | null;
    metrics: any;
  } {
    this.dataBuffer.push(values);

    if (this.dataBuffer.length < 10) {
      return { prediction: null, metrics: null };
    }

    // Keep sliding window of 10
    if (this.dataBuffer.length > 10) {
      this.dataBuffer.shift();
    }

    // Train on current window
    const x = this.dataBuffer.slice(0, 9).flat();
    const y = this.dataBuffer[9];

    const fitResult = this.model.fitOnline({
      xCoordinates: [x],
      yCoordinates: [y],
    });

    // Get next prediction
    const predictions = this.model.predict(1);

    return {
      prediction: predictions.isModelReady
        ? predictions.predictions[0].predicted
        : null,
      metrics: {
        loss: fitResult.loss,
        accuracy: predictions.accuracy,
        driftDetected: fitResult.driftDetected,
        learningRate: fitResult.effectiveLearningRate,
      },
    };
  }

  getStats() {
    return this.model.getModelSummary();
  }
}
```

---

## 🔬 Mathematical Background

### Loss Function

The model minimizes the regularized mean squared error:

$$L = \frac{1}{2n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|^2 + \frac{\lambda}{2}\sum_l\|W_l\|^2$$

Where:

- $n$ = batch size
- $y_i$ = target values
- $\hat{y}_i$ = predicted values
- $\lambda$ = regularization strength
- $W_l$ = weights at layer $l$

---

### Adam Optimizer

The Adam optimizer maintains running estimates of first and second moments:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

### Welford's Algorithm

For numerically stable online computation of mean and variance:

$$\delta = x - \mu_{n-1}$$

$$\mu_n = \mu_{n-1} + \frac{\delta}{n}$$

$$M_{2,n} = M_{2,n-1} + \delta(x - \mu_n)$$

$$\sigma^2 = \frac{M_2}{n-1}$$

---

### ADWIN Drift Detection

Detects distribution change when:

$$|\mu_0 - \mu_1| \geq \epsilon_{cut}$$

Where:

$$\epsilon_{cut} = \sqrt{\frac{2}{m} \ln\frac{2}{\delta}}$$

And $m = \frac{n_0 \cdot n_1}{n_0 + n_1}$ (harmonic mean of window sizes)

---

## 🎯 Use Cases

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RECOMMENDED USE CASES                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  📈 FINANCIAL                  🏭 INDUSTRIAL                           │
│  ├─ Stock price prediction     ├─ Equipment monitoring                 │
│  ├─ Currency exchange rates    ├─ Predictive maintenance               │
│  └─ Risk assessment            └─ Quality control                      │
│                                                                        │
│  🌤️ ENVIRONMENTAL              🏥 HEALTHCARE                           │
│  ├─ Weather forecasting        ├─ Patient monitoring                   │
│  ├─ Air quality prediction     ├─ Vital signs prediction               │
│  └─ Energy demand              └─ Epidemic modeling                    │
│                                                                        │
│  🌐 IoT/EDGE                   📊 ANALYTICS                            │
│  ├─ Sensor data prediction     ├─ User behavior prediction             │
│  ├─ Anomaly detection          ├─ Demand forecasting                   │
│  └─ Real-time adaptation       └─ Trend analysis                       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ❓ FAQ

<details>
<summary><strong>Q: How much data do I need to train the model?</strong></summary>

The model can start making predictions after just one sample due to its online
learning design. However, for reliable predictions:

- **Minimum**: 10-20 samples
- **Recommended**: 100+ samples
- **Optimal**: 1000+ samples

The model continuously improves as it sees more data.

</details>

<details>
<summary><strong>Q: Can I use this for classification?</strong></summary>

This library is specifically designed for **regression** tasks (predicting
continuous values). For classification, you would need to:

1. Modify the output layer to use softmax activation
2. Change the loss function to cross-entropy
3. Post-process outputs as probabilities

Consider using a dedicated classification library instead.

</details>

<details>
<summary><strong>Q: How do I handle missing data?</strong></summary>

The library doesn't have built-in missing data handling. Recommended approaches:

1. **Imputation**: Fill missing values with mean/median
2. **Interpolation**: Use linear/spline interpolation
3. **Skip**: Exclude samples with missing values
4. **Masking**: Replace with a special value (e.g., 0) and rely on normalization

```typescript
// Example: Mean imputation
const fillMissing = (data: number[], mean: number) =>
  data.map((v) => isNaN(v) ? mean : v);
```

</details>

<details>
<summary><strong>Q: What's the difference between `fitOnline` and batch training?</strong></summary>

| Aspect     | `fitOnline` (This Library) | Traditional Batch   |
| ---------- | -------------------------- | ------------------- |
| Memory     | O(1) per sample            | O(n) for all data   |
| Adaptation | Continuous                 | After retraining    |
| Speed      | Instant updates            | Full epoch required |
| Use Case   | Streaming data             | Static datasets     |

</details>

<details>
<summary><strong>Q: How do I know if my model is overfitting?</strong></summary>

Signs of overfitting:

1. Very low training loss but poor predictions
2. Large gap between training and validation loss
3. Predictions that are too "confident" (narrow confidence intervals)

Solutions:

- Increase `regularizationStrength`
- Reduce `hiddenLayers` or `convolutionsPerLayer`
- Collect more training data
- Enable stronger outlier detection

</details>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - Henrique Emanoel Viana, 2025

---

<div align="center">

```
═══════════════════════════════════════════════════════════════════
   Made with ❤️ for the machine learning community
   
   ⭐ Star this repo if you find it useful!
═══════════════════════════════════════════════════════════════════
```

**[Back to Top](#-convolutionalregression)**

</div>
