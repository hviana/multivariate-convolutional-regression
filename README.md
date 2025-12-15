> ## ⚠️🚨 IMPORTANT: THIS LIBRARY HAS BEEN DEPRECATED 🚨⚠️
> 
> ---
> 
> ### 🔄 This library has been replaced by a newer, more powerful version!
> 
> <table>
> <tr>
> <td>
> 
> ### ❌ OLD (This Repository)
> `@hviana/multivariate-convolutional-regression`
> 
> </td>
> <td>
> 
> ### ✅ NEW (Use This Instead)
> `@hviana/multivariate-regression`
> 
> </td>
> </tr>
> </table>
> 
> ---
> 
> ### 📦 Migration Links
> 
> | Platform | Link |
> |----------|------|
> | 🌐 **JSR Registry** | 👉 [https://jsr.io/@hviana/multivariate-regression](https://jsr.io/@hviana/multivariate-regression) |
> | 🐙 **GitHub Repository** | 👉 [https://github.com/hviana/multivariate-regression](https://github.com/hviana/multivariate-regression) |
> 
> ---
> 
> ### 🛑 Please migrate to the new library for:
> - ✨ New features and improvements
> - 🐛 Bug fixes and security updates
> - 📚 Better documentation
> - 🔧 Continued maintenance and support
> 
> ---

Model: # 📊 Multivariate Convolutional Regression

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Runtime](https://img.shields.io/badge/runtime-Deno%20%7C%20Node.js-yellow.svg)
![Year](https://img.shields.io/badge/year-2025-purple.svg)

**A high-performance Temporal Convolutional Network (TCN) for multivariate time
series regression with incremental online learning**

[📦 JSR Package](https://jsr.io/@hviana/multivariate-convolutional-regression) •
[🐙 GitHub](https://github.com/hviana/multivariate-convolutional-regression) •
[📖 Documentation](#-api-reference)

---

_Created by **Henrique Emanoel Viana**_

</div>

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🔧 Configuration Reference](#-configuration-reference)
- [📚 Concepts & Theory](#-concepts--theory)
- [🎯 Parameter Optimization Guide](#-parameter-optimization-guide)
- [💡 Use Case Examples](#-use-case-examples)
- [📖 API Reference](#-api-reference)
- [⚡ Performance Tips](#-performance-tips)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 **Advanced Architecture**

- ✅ Temporal Convolutional Networks (TCN)
- ✅ Causal dilated convolutions
- ✅ Residual connections
- ✅ Multi-horizon predictions
- ✅ Configurable depth & width

</td>
<td width="50%">

### 📈 **Online Learning**

- ✅ Incremental training (sample-by-sample)
- ✅ Adam optimizer with bias correction
- ✅ Welford z-score normalization
- ✅ ADWIN concept drift detection
- ✅ Outlier-aware sample weighting

</td>
</tr>
<tr>
<td width="50%">

### 🎯 **Prediction Quality**

- ✅ Uncertainty quantification
- ✅ Confidence bounds estimation
- ✅ Multi-step forecasting
- ✅ Automatic feature normalization

</td>
<td width="50%">

### ⚡ **Performance Optimized**

- ✅ Zero hot-path allocations
- ✅ Preallocated buffer pools
- ✅ Memory-efficient tensor operations
- ✅ CPU/memory constrained environments

</td>
</tr>
</table>

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TEMPORAL CONVOLUTIONAL NETWORK                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────────────────────────┐    ┌─────────┐ │
│  │   INPUT     │    │         TCN BACKBONE                │    │ OUTPUT  │ │
│  │  SEQUENCE   │──▶│  ┌───────┐ ┌───────┐     ┌───────┐  │──▶│  HEAD   │ │
│  │ [T×Features]│    │  │Block 1│─│Block 2│─···─│Block N│  │    │[Targets]│ │
│  └─────────────┘    │  │ d=1   │ │ d=2   │     │ d=2^n │  │    └─────────┘ │
│        │            │  └───────┘ └───────┘     └───────┘  │         │      │
│        │            └─────────────────────────────────────┘         │      │
│        │                                                            │      │
│        ▼                                                            ▼      │
│  ┌─────────────┐                                              ┌─────────┐  │
│  │  Welford    │                                              │Residual │  │
│  │Normalizer   │                                              │ Tracker │  │
│  └─────────────┘                                              └─────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 🔲 TCN Block Architecture

```
                 ┌──────────────────────────────────────────┐
                 │              TCN BLOCK                   │
                 │                                          │
Input ─────────▶│  ┌─────────┐    ┌─────────┐              │
  │              │  │ Causal  │    │  Causal │             │
  │              │  │ Conv1D  │──▶│  Conv1D │──┐          │
  │              │  │(dilated)│    │(dilated)│  │          │
  │              │  └─────────┘    └─────────┘  │          │
  │              │       │              │       │          │
  │              │       ▼              ▼       │          │
  │              │  ┌─────────┐    ┌─────────┐  │          │
  │              │  │  ReLU/  │    │  ReLU/  │  │          │
  │              │  │  GELU   │    │  GELU   │  │          │
  │              │  └─────────┘    └─────────┘  │          │
  │              │                      │       │          │
  │              │                      ▼       │          │
  │              │              ┌───────────────│          │
  │              │              │  LayerNorm   ││          │
  │              │              │  (optional)  ││          │
  │              │              └───────────────│          │
  │              │                      │       │          │
  │              │                      ▼       │          │
  │              │              ┌───────────────│          │
  │              │              │   Dropout    ││          │
  │              │              │  (training)  ││          │
  │              │              └───────────────│          │
  │              │                      │       │          │
  │              └──────────────────────│───────┘          │
  │                                     │                  │
  │         ┌─────────────────┐         │                  │
  └───────▶│ Residual Proj.  │─────────(+)─────────▶ Output
            │   (if needed)   │
            └─────────────────┘
```

### 📊 Receptive Field Visualization

```
Dilation Pattern (kernelSize=3, dilationBase=2, nBlocks=4):

Block 0 (d=1):   ■ ■ ■                    RF contribution: 2
Block 1 (d=2):   ■ · ■ · ■                RF contribution: 4
Block 2 (d=4):   ■ · · · ■ · · · ■        RF contribution: 8
Block 3 (d=8):   ■ · · · · · · · ■ · · · · · · · ■   RF contribution: 16

Total Receptive Field = 1 + 2×(2 + 4 + 8 + 16) = 61 timesteps
(with useTwoLayerBlock=true)
```

---

## 📦 Installation

### Deno

```typescript
import { TCNRegression } from "jsr:@hviana/multivariate-convolutional-regression";
```

### Node.js (via JSR)

```bash
npx jsr add @hviana/multivariate-convolutional-regression
```

```javascript
import { TCNRegression } from "@hviana/multivariate-convolutional-regression";
```

---

## 🚀 Quick Start

### Basic Example

```typescript
import { TCNRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// 1️⃣ Create the model
const tcn = new TCNRegression({
  maxSequenceLength: 64, // Lookback window
  maxFutureSteps: 5, // Prediction horizon
  hiddenChannels: 32, // Model capacity
  nBlocks: 4, // Network depth
});

// 2️⃣ Train incrementally with streaming data
const trainingData = [
  { inputs: [1.0, 2.0, 3.0], outputs: [4.0, 5.0] },
  { inputs: [1.1, 2.1, 3.1], outputs: [4.1, 5.1] },
  // ... more samples
];

for (const sample of trainingData) {
  const result = tcn.fitOnline({
    xCoordinates: [sample.inputs], // [timesteps][features]
    yCoordinates: [sample.outputs], // [timesteps][targets]
  });

  console.log(
    `📉 Loss: ${result.loss.toFixed(4)} | Weight: ${
      result.sampleWeight.toFixed(2)
    }`,
  );

  if (result.driftDetected) {
    console.log("⚠️ Concept drift detected!");
  }
}

// 3️⃣ Generate predictions
const predictions = tcn.predict(5); // Predict 5 steps ahead

console.log("🔮 Predictions:", predictions.predictions);
console.log("📊 Confidence:", predictions.confidence);
console.log("📈 Upper bounds:", predictions.uncertaintyUpper);
console.log("📉 Lower bounds:", predictions.uncertaintyLower);
```

### Batch Training Example

```typescript
// Train with multiple timesteps at once
const batchResult = tcn.fitOnline({
  xCoordinates: [
    [1.0, 2.0, 3.0], // t=0
    [1.1, 2.1, 3.1], // t=1
    [1.2, 2.2, 3.2], // t=2
    [1.3, 2.3, 3.3], // t=3
  ],
  yCoordinates: [
    [4.0, 5.0], // t=0
    [4.1, 5.1], // t=1
    [4.2, 5.2], // t=2
    [4.3, 5.3], // t=3
  ],
});
```

---

## 🔧 Configuration Reference

### Complete Configuration Interface

```typescript
interface TCNRegressionConfig {
  // ═══════════════════════════════════════════════════════════
  // 🏗️ ARCHITECTURE PARAMETERS
  // ═══════════════════════════════════════════════════════════

  maxSequenceLength?: number; // Default: 64
  maxFutureSteps?: number; // Default: 1
  hiddenChannels?: number; // Default: 32
  nBlocks?: number; // Default: 4
  kernelSize?: number; // Default: 3
  dilationBase?: number; // Default: 2
  useTwoLayerBlock?: boolean; // Default: true
  activation?: "relu" | "gelu"; // Default: "relu"
  useLayerNorm?: boolean; // Default: false
  dropoutRate?: number; // Default: 0.0

  // ═══════════════════════════════════════════════════════════
  // 📈 OPTIMIZER PARAMETERS
  // ═══════════════════════════════════════════════════════════

  learningRate?: number; // Default: 0.001
  beta1?: number; // Default: 0.9
  beta2?: number; // Default: 0.999
  epsilon?: number; // Default: 1e-8
  l2Lambda?: number; // Default: 0.0001
  gradientClipNorm?: number; // Default: 1.0

  // ═══════════════════════════════════════════════════════════
  // 📊 NORMALIZATION PARAMETERS
  // ═══════════════════════════════════════════════════════════

  normalizationEpsilon?: number; // Default: 1e-8
  normalizationWarmup?: number; // Default: 10

  // ═══════════════════════════════════════════════════════════
  // 🛡️ ROBUSTNESS PARAMETERS
  // ═══════════════════════════════════════════════════════════

  outlierThreshold?: number; // Default: 3.0
  outlierMinWeight?: number; // Default: 0.1
  adwinEnabled?: boolean; // Default: true
  adwinDelta?: number; // Default: 0.002
  adwinMaxBuckets?: number; // Default: 64

  // ═══════════════════════════════════════════════════════════
  // 🎯 PREDICTION PARAMETERS
  // ═══════════════════════════════════════════════════════════

  useDirectMultiHorizon?: boolean; // Default: true
  residualWindowSize?: number; // Default: 100
  uncertaintyMultiplier?: number; // Default: 1.96

  // ═══════════════════════════════════════════════════════════
  // ⚙️ MISC PARAMETERS
  // ═══════════════════════════════════════════════════════════

  weightInitScale?: number; // Default: 0.1
  seed?: number; // Default: 42
  verbose?: boolean; // Default: false
}
```

---

## 📚 Concepts & Theory

### 🌊 Temporal Convolutional Networks (TCN)

TCNs are a class of neural networks designed for sequence modeling that use
**causal convolutions** to ensure that predictions at time `t` only depend on
data from time `t` and earlier.

```
Standard Convolution (non-causal):
   Past ◀──────────────────────────▶ Future
         ▓▓▓▓▓▓▓▓▓█▓▓▓▓▓▓▓▓▓
                  ▲
            Uses future data ❌

Causal Convolution:
   Past ◀──────────────────────────▶ Future
         ▓▓▓▓▓▓▓▓▓█
                  ▲
            Only past data ✅
```

### 🔄 Dilated Convolutions

Dilated convolutions allow the network to have a large **receptive field**
without increasing the number of parameters:

```
Regular (d=1):    ■ ■ ■         Receptive field: 3
Dilated (d=2):    ■ · ■ · ■     Receptive field: 5
Dilated (d=4):    ■ · · · ■ · · · ■   Receptive field: 9
```

**Receptive Field Formula:**

```
RF = 1 + Σ(layers_per_block × (kernel_size - 1) × dilation_i)
```

### 📐 Welford Online Normalization

The library uses **Welford's algorithm** for numerically stable online
computation of mean and variance:

```
For each new sample x:
  n = n + 1
  delta = x - mean
  mean = mean + delta / n
  M2 = M2 + delta × (x - mean)
  variance = M2 / (n - 1)
```

**Benefits:**

- ✅ Single-pass computation
- ✅ Numerically stable for large n
- ✅ No need to store all samples
- ✅ Handles streaming data

### 🚨 ADWIN Drift Detection

**ADWIN (ADaptive WINdowing)** detects concept drift by maintaining a
variable-length window and detecting significant changes in data distribution:

```
┌────────────────────────────────────────────────────┐
│              ADWIN Window                          │
├─────────────────────┬──────────────────────────────┤
│   Old Distribution  │      New Distribution        │
│      μ₁ = 5.2       │         μ₂ = 7.8             │
│      σ₁ = 1.1       │         σ₂ = 1.3             │
└─────────────────────┴──────────────────────────────┘
                      ▲
                      │
         |μ₁ - μ₂| > ε(δ, n₁, n₂) → DRIFT DETECTED!
```

### ⚖️ Outlier-Aware Sample Weighting

Samples with high prediction errors (potential outliers) are downweighted during
training:

```
z_score = |error| / error_std

weight = {
  1.0,                           if z_score ≤ threshold
  max(min_weight, threshold/z),  if z_score > threshold
}
```

---

## 🎯 Parameter Optimization Guide

### 🏗️ Architecture Parameters

#### `maxSequenceLength` (default: 64)

The maximum lookback window size. Determines how much historical data the model
can use.

| Value   | Use Case                                 | Memory Impact |
| ------- | ---------------------------------------- | ------------- |
| 16-32   | Short-term patterns, high-frequency data | Low           |
| 64-128  | Medium-term dependencies                 | Medium        |
| 256-512 | Long-term patterns, seasonal data        | High          |

```typescript
// 📊 Stock price prediction (minute data, short patterns)
const shortTerm = new TCNRegression({ maxSequenceLength: 32 });

// 🌡️ Temperature forecasting (daily data, seasonal patterns)
const seasonal = new TCNRegression({ maxSequenceLength: 365 });
```

---

#### `maxFutureSteps` (default: 1)

Maximum prediction horizon. How many steps ahead to forecast.

| Value | Use Case                                   |
| ----- | ------------------------------------------ |
| 1     | Single-step forecasting, real-time systems |
| 5-10  | Short-term planning                        |
| 20+   | Long-term forecasting (higher uncertainty) |

```typescript
// ⚡ Real-time anomaly detection
const realtime = new TCNRegression({ maxFutureSteps: 1 });

// 📈 Weekly sales forecasting (predict 7 days)
const weekly = new TCNRegression({ maxFutureSteps: 7 });
```

---

#### `hiddenChannels` (default: 32)

Number of channels in the TCN blocks. Controls model capacity.

| Value  | Use Case                     | Parameters |
| ------ | ---------------------------- | ---------- |
| 16     | Simple patterns, low compute | ~10K       |
| 32     | Balanced (default)           | ~40K       |
| 64-128 | Complex patterns             | ~150K+     |

```typescript
// 💡 Simple univariate regression
const simple = new TCNRegression({ hiddenChannels: 16 });

// 🔬 Complex multivariate with many features
const complex = new TCNRegression({ hiddenChannels: 128 });
```

---

#### `nBlocks` (default: 4)

Number of residual TCN blocks. Affects depth and receptive field.

```
Receptive Field ≈ 1 + nBlocks × 2 × (kernelSize - 1) × (dilationBase^nBlocks - 1) / (dilationBase - 1)
```

| nBlocks | Receptive Field (k=3, d=2) | Use Case            |
| ------- | -------------------------- | ------------------- |
| 2       | ~13                        | Very short patterns |
| 4       | ~61                        | Medium patterns     |
| 6       | ~253                       | Long dependencies   |
| 8       | ~1021                      | Very long sequences |

```typescript
// 🎵 Audio processing (needs large receptive field)
const audio = new TCNRegression({ nBlocks: 8, kernelSize: 3 });

// 📱 IoT sensor data (short patterns)
const iot = new TCNRegression({ nBlocks: 2 });
```

---

#### `kernelSize` (default: 3)

Convolution kernel size. Larger kernels capture more local context.

| Value | Characteristics                     |
| ----- | ----------------------------------- |
| 2     | Minimal, fastest                    |
| 3     | Standard, good balance              |
| 5-7   | More local context, more parameters |

```typescript
// ⚡ Minimal overhead
const fast = new TCNRegression({ kernelSize: 2 });

// 📊 Rich local patterns
const rich = new TCNRegression({ kernelSize: 5 });
```

---

#### `dilationBase` (default: 2)

Dilation growth factor. Controls how quickly receptive field expands.

| Value | Pattern                                          |
| ----- | ------------------------------------------------ |
| 2     | 1, 2, 4, 8, 16... (standard, exponential growth) |
| 3     | 1, 3, 9, 27... (faster growth, sparser coverage) |

```typescript
// Standard exponential dilation
const standard = new TCNRegression({ dilationBase: 2 });

// Faster receptive field growth
const fast = new TCNRegression({ dilationBase: 3, nBlocks: 3 });
```

---

#### `useTwoLayerBlock` (default: true)

Whether to use 2 convolutional layers per TCN block.

| Setting | Pros                              | Cons                  |
| ------- | --------------------------------- | --------------------- |
| `true`  | More expressive, better gradients | 2x parameters, slower |
| `false` | Faster, fewer parameters          | Less capacity         |

---

#### `activation` (default: "relu")

Activation function used in TCN blocks.

| Option   | Characteristics                                  |
| -------- | ------------------------------------------------ |
| `"relu"` | Fast, sparse activations, may cause dead neurons |
| `"gelu"` | Smoother, better gradients, slightly slower      |

```typescript
// 🚀 Fast training
const fast = new TCNRegression({ activation: "relu" });

// 🎯 Potentially better convergence
const smooth = new TCNRegression({ activation: "gelu" });
```

---

#### `useLayerNorm` (default: false)

Enable layer normalization in TCN blocks.

| Setting | Use Case                                       |
| ------- | ---------------------------------------------- |
| `false` | Faster, simpler, works well for many cases     |
| `true`  | Better for deep networks, varying input scales |

---

#### `dropoutRate` (default: 0.0)

Dropout probability during training for regularization.

| Value   | Use Case                                  |
| ------- | ----------------------------------------- |
| 0.0     | Online learning (default), small datasets |
| 0.1-0.3 | Large datasets, overfitting prevention    |
| 0.5     | Heavy regularization                      |

---

### 📈 Optimizer Parameters

#### `learningRate` (default: 0.001)

Adam optimizer learning rate.

| Value  | Use Case                         |
| ------ | -------------------------------- |
| 0.0001 | Stable, slow convergence         |
| 0.001  | Default, good balance            |
| 0.01   | Fast adaptation, may be unstable |

```typescript
// 🐢 Stable training for noisy data
const stable = new TCNRegression({ learningRate: 0.0001 });

// 🐇 Fast adaptation for clean data
const fast = new TCNRegression({ learningRate: 0.01 });
```

---

#### `beta1` / `beta2` (defaults: 0.9 / 0.999)

Adam momentum parameters.

| Parameter | Default | Effect of Higher Value          |
| --------- | ------- | ------------------------------- |
| `beta1`   | 0.9     | More momentum, smoother updates |
| `beta2`   | 0.999   | Slower learning rate adaptation |

---

#### `l2Lambda` (default: 0.0001)

L2 regularization coefficient (weight decay).

| Value      | Use Case                       |
| ---------- | ------------------------------ |
| 0          | No regularization              |
| 0.0001     | Light regularization (default) |
| 0.001-0.01 | Strong regularization          |

---

#### `gradientClipNorm` (default: 1.0)

Maximum gradient L2 norm for gradient clipping.

| Value    | Use Case                  |
| -------- | ------------------------- |
| 0.1-0.5  | Very conservative, stable |
| 1.0      | Default, good balance     |
| 5.0-10.0 | Allow larger updates      |

---

### 🛡️ Robustness Parameters

#### `outlierThreshold` (default: 3.0)

Z-score threshold for downweighting outliers.

```
Sample weight = 1.0 if |error|/std ≤ threshold
Sample weight = threshold / z_score otherwise
```

| Value | Behavior                     |
| ----- | ---------------------------- |
| 2.0   | Aggressive outlier detection |
| 3.0   | Standard (default)           |
| 5.0   | Only extreme outliers        |

---

#### `adwinEnabled` (default: true)

Enable ADWIN concept drift detection.

```typescript
// Detect data distribution changes
const withDrift = new TCNRegression({ adwinEnabled: true });

// Stable environments
const stable = new TCNRegression({ adwinEnabled: false });
```

---

#### `adwinDelta` (default: 0.002)

ADWIN significance level. Lower = more sensitive to drift.

| Value  | Sensitivity    |
| ------ | -------------- |
| 0.0001 | Very sensitive |
| 0.002  | Default        |
| 0.01   | Less sensitive |

---

### 🎯 Prediction Parameters

#### `useDirectMultiHorizon` (default: true)

Direct vs recursive multi-step prediction.

| Mode                | Mechanism                              | Pros                          | Cons                     |
| ------------------- | -------------------------------------- | ----------------------------- | ------------------------ |
| `true` (Direct)     | Single forward pass predicts all steps | Faster, no error accumulation | Requires more parameters |
| `false` (Recursive) | Iteratively predict one step at a time | Fewer parameters              | Error accumulation       |

---

#### `residualWindowSize` (default: 100)

Window size for tracking prediction residuals (used for uncertainty estimation).

| Value | Uncertainty Estimates       |
| ----- | --------------------------- |
| 50    | Responsive, higher variance |
| 100   | Balanced (default)          |
| 500   | Stable, slower adaptation   |

---

#### `uncertaintyMultiplier` (default: 1.96)

Z-multiplier for confidence bounds.

| Value | Confidence Level |
| ----- | ---------------- |
| 1.0   | ~68%             |
| 1.64  | ~90%             |
| 1.96  | ~95% (default)   |
| 2.58  | ~99%             |

---

## 💡 Use Case Examples

### 📈 Financial Time Series

```typescript
// Stock price prediction with high-frequency data
const financialModel = new TCNRegression({
  // Short lookback, prices change quickly
  maxSequenceLength: 32,
  maxFutureSteps: 5,

  // Higher capacity for complex patterns
  hiddenChannels: 64,
  nBlocks: 4,

  // Smooth activation for continuous values
  activation: "gelu",

  // Handle volatile markets
  outlierThreshold: 2.5,
  adwinEnabled: true,
  adwinDelta: 0.001, // Sensitive to regime changes

  // Conservative learning
  learningRate: 0.0005,
  l2Lambda: 0.001,

  // Wide confidence intervals for risk management
  uncertaintyMultiplier: 2.58, // 99% confidence
});

// Training loop
for (const candle of marketData) {
  const result = financialModel.fitOnline({
    xCoordinates: [[candle.open, candle.high, candle.low, candle.volume]],
    yCoordinates: [[candle.close]],
  });

  if (result.driftDetected) {
    console.log("🚨 Market regime change detected!");
  }
}

// Get predictions with risk bounds
const forecast = financialModel.predict(5);
console.log("Expected prices:", forecast.predictions);
console.log("Worst case:", forecast.uncertaintyLower);
console.log("Best case:", forecast.uncertaintyUpper);
```

---

### 🌡️ IoT Sensor Monitoring

```typescript
// Temperature and humidity prediction from sensors
const sensorModel = new TCNRegression({
  // Moderate lookback for environmental data
  maxSequenceLength: 96, // 4 days at 1-hour intervals
  maxFutureSteps: 24, // 24-hour forecast

  // Simpler model for periodic patterns
  hiddenChannels: 32,
  nBlocks: 3,
  kernelSize: 3,

  // Standard settings
  activation: "relu",
  dropoutRate: 0.1,

  // Robust to sensor noise
  outlierThreshold: 3.5,
  normalizationWarmup: 24,

  // Drift detection for sensor failures
  adwinEnabled: true,
});

// Real-time sensor streaming
sensorSocket.on("data", (reading) => {
  const result = sensorModel.fitOnline({
    xCoordinates: [[reading.temp, reading.humidity, reading.pressure]],
    yCoordinates: [[reading.temp, reading.humidity]],
  });

  // Anomaly detection
  if (result.sampleWeight < 0.5) {
    console.log("⚠️ Unusual sensor reading detected!");
  }

  // Update dashboard predictions
  const forecast = sensorModel.predict(24);
  updateDashboard(forecast);
});
```

---

### 🏭 Industrial Process Control

```typescript
// Multi-input multi-output process prediction
const processModel = new TCNRegression({
  // Long history for complex processes
  maxSequenceLength: 128,
  maxFutureSteps: 10,

  // High capacity for many variables
  hiddenChannels: 128,
  nBlocks: 6,
  useTwoLayerBlock: true,

  // Best gradients for deep network
  activation: "gelu",
  useLayerNorm: true,

  // Strong regularization
  l2Lambda: 0.001,
  dropoutRate: 0.2,
  gradientClipNorm: 0.5,

  // Tight control bounds
  uncertaintyMultiplier: 1.96,
});

// Training with multiple process variables
processModel.fitOnline({
  xCoordinates: processHistory.map((t) => [
    t.temperature,
    t.pressure,
    t.flowRate,
    t.concentration,
    t.setpoint,
  ]),
  yCoordinates: processHistory.map((t) => [
    t.output_quality,
    t.yield,
    t.energy_consumption,
  ]),
});
```

---

### 📊 Demand Forecasting

```typescript
// E-commerce demand prediction
const demandModel = new TCNRegression({
  // Weekly patterns with seasonal context
  maxSequenceLength: 365, // One year of daily data
  maxFutureSteps: 30, // 30-day forecast

  // Moderate complexity
  hiddenChannels: 48,
  nBlocks: 5,
  dilationBase: 2, // Standard dilation

  // GELU for smooth predictions
  activation: "gelu",

  // Handle promotional spikes
  outlierThreshold: 4.0,
  outlierMinWeight: 0.3, // Don't ignore promotions entirely

  // Detect trend changes
  adwinEnabled: true,
  adwinDelta: 0.005,

  // Inventory planning confidence
  uncertaintyMultiplier: 1.64, // 90% confidence
  residualWindowSize: 200,
});

// Batch training with historical data
const batchSize = 30; // 30 days at a time
for (let i = 0; i < historicalData.length; i += batchSize) {
  const batch = historicalData.slice(i, i + batchSize);

  demandModel.fitOnline({
    xCoordinates: batch.map((d) => [
      d.dayOfWeek,
      d.monthOfYear,
      d.isHoliday ? 1 : 0,
      d.promotionActive ? 1 : 0,
      d.previousDaySales,
    ]),
    yCoordinates: batch.map((d) => [d.sales]),
  });
}

// Generate forecast
const forecast = demandModel.predict(30);

// Calculate safety stock from uncertainty
const safetyStock = forecast.uncertaintyUpper.map((upper, i) =>
  upper[0] - forecast.predictions[i][0]
);
```

---

## 📖 API Reference

### Class: `TCNRegression`

#### Constructor

```typescript
constructor(config?: TCNRegressionConfig)
```

Creates a new TCN regression model. Model initialization is deferred until first
`fitOnline()` call.

---

#### `fitOnline(input)`

```typescript
fitOnline(input: {
  xCoordinates: number[][];  // [timesteps][features]
  yCoordinates: number[][];  // [timesteps][targets]
}): FitResult
```

Performs incremental online training with one or more timesteps.

**Returns:**

```typescript
interface FitResult {
  loss: number; // Weighted MSE loss
  sampleWeight: number; // Average sample weight (outlier-adjusted)
  driftDetected: boolean; // Whether ADWIN detected drift
  metrics: {
    avgLoss: number; // Running average loss
    mae: number; // Mean absolute error
    sampleCount: number; // Total samples processed
  };
}
```

---

#### `predict(futureSteps)`

```typescript
predict(futureSteps: number): PredictionResult
```

Generates predictions for future timesteps.

**Returns:**

```typescript
interface PredictionResult {
  predictions: number[][]; // [futureSteps][targets]
  uncertaintyLower: number[][]; // Lower confidence bounds
  uncertaintyUpper: number[][]; // Upper confidence bounds
  confidence: number; // Overall confidence score (0-1)
}
```

---

#### `getModelSummary()`

```typescript
getModelSummary(): ModelSummary
```

Returns detailed model architecture information.

**Returns:**

```typescript
interface ModelSummary {
  architecture: string; // Human-readable summary
  totalParameters: number; // Total trainable parameters
  layerParameters: { [key: string]: number }; // Per-layer counts
  receptiveField: number; // Effective lookback
  memoryUsageBytes: number; // Estimated memory usage
  config: Required<TCNRegressionConfig>; // Full configuration
}
```

---

#### `getWeights()`

```typescript
getWeights(): WeightInfo
```

Returns all model weights organized by layer.

---

#### `getNormalizationStats()`

```typescript
getNormalizationStats(): NormalizationStats
```

Returns current normalization statistics.

**Returns:**

```typescript
interface NormalizationStats {
  inputMeans: number[];
  inputStds: number[];
  outputMeans: number[];
  outputStds: number[];
  sampleCount: number;
  warmupComplete: boolean;
}
```

---

#### `reset()`

```typescript
reset(): void
```

Resets model to initial state (re-initializes weights, clears history).

---

#### `save()`

```typescript
save(): string
```

Serializes complete model state to JSON string.

---

#### `load(jsonStr)`

```typescript
load(jsonStr: string): void
```

Restores model state from JSON string.

---

## ⚡ Performance Tips

### 🎯 Memory Optimization

```typescript
// Minimize memory for constrained environments
const memoryOptimized = new TCNRegression({
  maxSequenceLength: 32, // Smaller window
  hiddenChannels: 16, // Fewer channels
  nBlocks: 2, // Fewer blocks
  useTwoLayerBlock: false, // Single layer blocks
  residualWindowSize: 50, // Smaller residual buffer
});
```

### 🚀 Speed Optimization

```typescript
// Maximize throughput
const speedOptimized = new TCNRegression({
  activation: "relu", // Faster than GELU
  useLayerNorm: false, // Skip normalization
  useTwoLayerBlock: false, // Half the convolutions
  adwinEnabled: false, // Skip drift detection
  kernelSize: 2, // Minimal kernel
});
```

### 📊 Accuracy Optimization

```typescript
// Maximize prediction quality
const accuracyOptimized = new TCNRegression({
  hiddenChannels: 128, // High capacity
  nBlocks: 6, // Deep network
  useTwoLayerBlock: true, // Rich feature extraction
  activation: "gelu", // Smooth gradients
  useLayerNorm: true, // Stable training
  maxSequenceLength: 256, // Long context
  residualWindowSize: 500, // Stable uncertainty
});
```

### 🔄 Online Learning Best Practices

1. **Warm-up period**: Allow `normalizationWarmup` samples before relying on
   predictions
2. **Monitor drift**: Watch `driftDetected` flag for distribution changes
3. **Check confidence**: Use `confidence` score to gate critical decisions
4. **Save checkpoints**: Periodically call `save()` for recovery

```typescript
// Robust online learning loop
let bestLoss = Infinity;

for (const sample of stream) {
  const result = model.fitOnline({
    xCoordinates: [sample.x],
    yCoordinates: [sample.y],
  });

  // Handle drift
  if (result.driftDetected) {
    console.log("Drift detected, adapting...");
    // Optionally reset or adjust learning rate
  }

  // Checkpoint on improvement
  if (result.metrics.avgLoss < bestLoss) {
    bestLoss = result.metrics.avgLoss;
    checkpoint = model.save();
  }

  // Make predictions only when confident
  const pred = model.predict(1);
  if (pred.confidence > 0.7) {
    executePrediction(pred.predictions[0]);
  }
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests
on [GitHub](https://github.com/hviana/multivariate-convolutional-regression).

### Development Setup

```bash
# Clone the repository
git clone https://github.com/hviana/multivariate-convolutional-regression.git
cd multivariate-convolutional-regression

# Run tests (Deno)
deno test

# Format code
deno fmt

# Lint
deno lint
```

---

## 📄 License

MIT License © 2025 Henrique Emanoel Viana

---

<div align="center">

**[⬆ Back to Top](#-multivariate-convolutional-regression)**

Made with ❤️ for the time series community

</div>
