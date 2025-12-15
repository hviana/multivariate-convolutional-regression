# 📊 Multivariate Convolutional Regression

<div align="center">

**A powerful Temporal Convolutional Network (TCN) library for multivariate time
series regression with online learning capabilities**

[📦 JSR Package](https://jsr.io/@hviana/multivariate-convolutional-regression) •
[🐙 GitHub](https://github.com/hviana/multivariate-convolutional-regression) •
[📖 Documentation](#-table-of-contents)

</div>

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🚀 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [📚 API Reference](#-api-reference)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
- [🎯 Optimization Guide](#-optimization-guide)
- [📈 Use Case Examples](#-use-case-examples)
- [🧮 Mathematical Foundations](#-mathematical-foundations)
- [💾 Serialization](#-serialization)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Deep Learning Architecture

- **Temporal Convolutional Networks** with dilated causal convolutions
- **Residual connections** for gradient flow
- **Multi-horizon prediction** (direct or recursive)
- Configurable **activation functions** (ReLU/GELU)
- Optional **Layer Normalization**

</td>
<td width="50%">

### 📡 Online Learning

- **Single-sample training** (streaming data)
- **Welford algorithm** for running statistics
- **ADWIN drift detection** for concept drift
- **Outlier downweighting** for robustness
- No mini-batching required

</td>
</tr>
<tr>
<td width="50%">

### 🎛️ Advanced Optimization

- **Adam optimizer** with bias correction
- **L2 regularization** (weight decay)
- **Gradient clipping** by norm
- Automatic **z-score normalization**
- Xavier/He weight initialization

</td>
<td width="50%">

### 📊 Uncertainty Quantification

- **Prediction confidence intervals**
- **Residual-based uncertainty** estimation
- Configurable **confidence multiplier**
- Growing uncertainty for longer horizons

</td>
</tr>
</table>

### 🎯 Key Highlights

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ✅ Zero Dependencies        │  ✅ TypeScript Native    │  ✅ Memory Efficient │
│  ✅ Streaming Compatible     │  ✅ Auto-normalization   │  ✅ Drift Detection  │
│  ✅ Serializable             │  ✅ Multi-target Support │  ✅ Configurable     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Deno

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";
```

### Node.js (via JSR)

```bash
npx jsr add @hviana/multivariate-convolutional-regression
```

```typescript
import { ConvolutionalRegression } from "@hviana/multivariate-convolutional-regression";
```

---

## ⚡ Quick Start

### Basic Example

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// 🔧 Create model with default configuration
const model = new ConvolutionalRegression({
  maxSequenceLength: 32,
  hiddenChannels: 16,
  nBlocks: 3,
});

// 📊 Training data: predict y from x over time
const trainingData = [
  { x: [1.0, 2.0], y: [0.5] },
  { x: [1.2, 2.1], y: [0.52] },
  { x: [1.4, 2.3], y: [0.55] },
  // ... more timesteps
];

// 🎯 Train online (one sample at a time)
for (const sample of trainingData) {
  const result = model.fitOnline({
    xCoordinates: [sample.x],
    yCoordinates: [sample.y],
  });

  console.log(`📉 Loss: ${result.loss.toFixed(4)}`);
}

// 🔮 Make predictions
const prediction = model.predict(3); // Predict 3 steps ahead
console.log("Predictions:", prediction.predictions);
console.log("Confidence:", prediction.confidence);
```

---

## 🏗️ Architecture

### Network Overview

```
                           TCN ARCHITECTURE DIAGRAM
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   INPUT SEQUENCE                    TCN BACKBONE                   OUTPUT   │
│   [T × Features]                   (Residual Blocks)              [Targets] │
│                                                                             │
│   ┌───┬───┬───┐    ┌─────────────────────────────────────┐    ┌─────────┐  │
│   │ t │ t │ t │    │  ┌─────────┐   ┌─────────┐          │    │         │  │
│   │ 1 │ 2 │...│───▶│  │ Block 1 │──▶│ Block 2 │──▶ ... ──│───▶│  Head   │  │
│   │   │   │   │    │  │ d=1     │   │ d=2     │   d=2^n  │    │         │  │
│   └───┴───┴───┘    │  └────┬────┘   └────┬────┘          │    └────┬────┘  │
│                    │       │             │               │         │       │
│                    │       └─────────────┴───────────────│         ▼       │
│                    │           (Residual Connections)    │   ┌─────────┐   │
│                    └─────────────────────────────────────┘   │y₁ y₂ ...│   │
│                                                              └─────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### TCN Block Detail

```
                              TCN RESIDUAL BLOCK
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│     Input                                                                │
│       │                                                                  │
│       ├─────────────────────────────────────────────┐                    │
│       │                                             │                    │
│       ▼                                             │ (Residual)         │
│  ┌─────────────┐                                    │                    │
│  │ Causal Conv │  Dilated, kernel_size=k            │                    │
│  └──────┬──────┘                                    │                    │
│         │                                           │                    │
│         ▼                                           │                    │
│  ┌─────────────┐                                    │                    │
│  │ Activation  │  ReLU or GELU                      │                    │
│  └──────┬──────┘                                    │                    │
│         │                                           │                    │
│         ▼ (if useTwoLayerBlock)                     │                    │
│  ┌─────────────┐                                    │                    │
│  │ Causal Conv │                                    │                    │
│  └──────┬──────┘                                    │                    │
│         │                                           │                    │
│         ▼                                           ▼                    │
│  ┌─────────────┐                            ┌─────────────┐              │
│  │ Activation  │                            │  1×1 Conv   │ (if needed)  │
│  └──────┬──────┘                            └──────┬──────┘              │
│         │                                          │                    │
│         ▼ (optional)                               │                    │
│  ┌─────────────┐                                   │                    │
│  │ Layer Norm  │                                   │                    │
│  └──────┬──────┘                                   │                    │
│         │                                          │                    │
│         ▼ (optional)                               │                    │
│  ┌─────────────┐                                   │                    │
│  │   Dropout   │                                   │                    │
│  └──────┬──────┘                                   │                    │
│         │                                          │                    │
│         └──────────────────┬───────────────────────┘                    │
│                            │                                            │
│                            ▼                                            │
│                     ┌─────────────┐                                     │
│                     │     ADD     │                                     │
│                     └──────┬──────┘                                     │
│                            │                                            │
│                            ▼                                            │
│                         Output                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Causal Dilated Convolution

```
    RECEPTIVE FIELD WITH DILATION
    
Dilation = 1    Dilation = 2    Dilation = 4

     ●               ●               ●          Output
    /|\             /|\             /|\
   / | \           / | \           / | \
  ●  ●  ●         ●  ●  ●         ●  ●  ●       Hidden
  │  │  │         │     │         │           │
  │  │  │         │     │         │           │
  ●  ●  ●  ●      ●  ●  ●  ●      ●  ●  ●  ●  ●  ●  ●  ●  Input
  t-2 t-1 t       t-4 t-2 t       t-8 t-4 t
  
Receptive Field = Σ(kernel_size - 1) × dilation + 1
```

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw Input    Normalize     Ring Buffer    TCN Forward    Denormalize      │
│  ─────────▶  ───────────▶  ───────────▶  ────────────▶  ────────────▶     │
│   [x, y]       z-score      Sequence       Features       Predictions      │
│              (Welford)      History        Extraction                       │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     TRAINING PATH (fitOnline)                        │   │
│  │                                                                      │   │
│  │   Compute Loss ──▶ Outlier Weight ──▶ Backward Pass ──▶ Adam Step   │   │
│  │        │                                      │             │        │   │
│  │        ▼                                      ▼             ▼        │   │
│  │   ADWIN Drift                          Gradient Clip    Update       │   │
│  │   Detection                            by Norm          Moments      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 API Reference

### Constructor

```typescript
const model = new ConvolutionalRegression(config?: TCNRegressionConfig);
```

### Methods

| Method                    | Description             | Returns              |
| ------------------------- | ----------------------- | -------------------- |
| `fitOnline(data)`         | Train on single sample  | `FitResult`          |
| `predict(futureSteps?)`   | Generate predictions    | `PredictionResult`   |
| `getModelSummary()`       | Get architecture info   | `ModelSummary`       |
| `getWeights()`            | Inspect parameters      | `WeightInfo`         |
| `getNormalizationStats()` | Get normalization state | `NormalizationStats` |
| `reset()`                 | Reset to initial state  | `void`               |
| `save()`                  | Serialize model         | `string`             |
| `load(json)`              | Deserialize model       | `void`               |

### Type Definitions

#### FitResult

```typescript
interface FitResult {
  loss: number; // MSE loss for this sample
  sampleWeight: number; // Applied sample weight (outlier handling)
  driftDetected: boolean; // Whether ADWIN detected drift
  metrics: {
    avgLoss: number; // Running average loss
    mae: number; // Mean absolute error
    sampleCount: number; // Total samples processed
  };
}
```

#### PredictionResult

```typescript
interface PredictionResult {
  predictions: number[][]; // [futureSteps][nTargets]
  uncertaintyLower: number[][]; // Lower confidence bounds
  uncertaintyUpper: number[][]; // Upper confidence bounds
  confidence: number; // 0-1 confidence score
}
```

#### ModelSummary

```typescript
interface ModelSummary {
  architecture: string; // Human-readable description
  layerParams: { [name: string]: number };
  totalParams: number;
  receptiveField: number; // Timesteps the model can "see"
  memoryBytes: number; // Estimated memory usage
  nFeatures: number;
  nTargets: number;
  sampleCount: number;
}
```

---

## ⚙️ Configuration Parameters

### 🏛️ Architecture Parameters

<table>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
<tr>
<td><code>maxSequenceLength</code></td>
<td>number</td>
<td>64</td>
<td>Maximum lookback window (receptive field cap). Determines how many past timesteps the model can consider.</td>
</tr>
<tr>
<td><code>maxFutureSteps</code></td>
<td>number</td>
<td>1</td>
<td>Maximum prediction horizon. Set higher for multi-step forecasting.</td>
</tr>
<tr>
<td><code>hiddenChannels</code></td>
<td>number</td>
<td>32</td>
<td>Number of channels in TCN blocks. Higher = more capacity, more compute.</td>
</tr>
<tr>
<td><code>nBlocks</code></td>
<td>number</td>
<td>4</td>
<td>Number of residual TCN blocks. More blocks = larger receptive field.</td>
</tr>
<tr>
<td><code>kernelSize</code></td>
<td>number</td>
<td>3</td>
<td>Convolution kernel size. Larger kernels capture longer local patterns.</td>
</tr>
<tr>
<td><code>dilationBase</code></td>
<td>number</td>
<td>2</td>
<td>Dilation growth factor. Dilations = base^blockIndex (1, 2, 4, 8...).</td>
</tr>
<tr>
<td><code>useTwoLayerBlock</code></td>
<td>boolean</td>
<td>true</td>
<td>Use 2 conv layers per TCN block for increased expressiveness.</td>
</tr>
<tr>
<td><code>useDirectMultiHorizon</code></td>
<td>boolean</td>
<td>true</td>
<td>Direct multi-step prediction vs recursive rollforward.</td>
</tr>
</table>

#### 📐 Receptive Field Formula

```
Receptive Field = 1 + Σᵢ (kernelSize - 1) × dilationBase^i × layersPerBlock

Example (defaults):
  nBlocks=4, kernelSize=3, dilationBase=2, useTwoLayerBlock=true
  RF = 1 + 2×(2×1 + 2×2 + 2×4 + 2×8) = 1 + 2×30 = 61 timesteps
```

### 🎛️ Activation & Normalization

<table>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
<tr>
<td><code>activation</code></td>
<td>"relu" | "gelu"</td>
<td>"relu"</td>
<td>Activation function. GELU is smoother but more expensive.</td>
</tr>
<tr>
<td><code>useLayerNorm</code></td>
<td>boolean</td>
<td>false</td>
<td>Enable channel normalization after convolutions. Helps with deep networks.</td>
</tr>
<tr>
<td><code>dropoutRate</code></td>
<td>number</td>
<td>0.0</td>
<td>Dropout probability during training (0.0 - 1.0).</td>
</tr>
</table>

### 📈 Optimizer Parameters

<table>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
<tr>
<td><code>learningRate</code></td>
<td>number</td>
<td>0.001</td>
<td>Adam learning rate. Lower for stability, higher for speed.</td>
</tr>
<tr>
<td><code>beta1</code></td>
<td>number</td>
<td>0.9</td>
<td>Adam first moment decay (momentum).</td>
</tr>
<tr>
<td><code>beta2</code></td>
<td>number</td>
<td>0.999</td>
<td>Adam second moment decay (adaptive learning).</td>
</tr>
<tr>
<td><code>epsilon</code></td>
<td>number</td>
<td>1e-8</td>
<td>Adam numerical stability constant.</td>
</tr>
<tr>
<td><code>l2Lambda</code></td>
<td>number</td>
<td>0.0001</td>
<td>L2 regularization coefficient (weight decay).</td>
</tr>
<tr>
<td><code>gradientClipNorm</code></td>
<td>number</td>
<td>1.0</td>
<td>Maximum gradient L2 norm to prevent exploding gradients.</td>
</tr>
</table>

### 📊 Normalization Parameters

<table>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
<tr>
<td><code>normalizationEpsilon</code></td>
<td>number</td>
<td>1e-8</td>
<td>Variance floor for numerical stability.</td>
</tr>
<tr>
<td><code>normalizationWarmup</code></td>
<td>number</td>
<td>10</td>
<td>Samples before applying z-score normalization.</td>
</tr>
</table>

### 🛡️ Robustness Parameters

<table>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
<tr>
<td><code>outlierThreshold</code></td>
<td>number</td>
<td>3.0</td>
<td>Z-score threshold for outlier detection.</td>
</tr>
<tr>
<td><code>outlierMinWeight</code></td>
<td>number</td>
<td>0.1</td>
<td>Minimum sample weight for outliers (0.0-1.0).</td>
</tr>
<tr>
<td><code>adwinEnabled</code></td>
<td>boolean</td>
<td>true</td>
<td>Enable ADWIN drift detection.</td>
</tr>
<tr>
<td><code>adwinDelta</code></td>
<td>number</td>
<td>0.002</td>
<td>ADWIN significance parameter. Lower = more sensitive.</td>
</tr>
<tr>
<td><code>adwinMaxBuckets</code></td>
<td>number</td>
<td>64</td>
<td>Maximum ADWIN bucket count (memory limit).</td>
</tr>
</table>

### 📉 Uncertainty Parameters

<table>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
<tr>
<td><code>residualWindowSize</code></td>
<td>number</td>
<td>100</td>
<td>Number of recent residuals for uncertainty estimation.</td>
</tr>
<tr>
<td><code>uncertaintyMultiplier</code></td>
<td>number</td>
<td>1.96</td>
<td>Z-multiplier for confidence bounds (1.96 ≈ 95%).</td>
</tr>
</table>

### 🔧 Misc Parameters

<table>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
<tr>
<td><code>weightInitScale</code></td>
<td>number</td>
<td>0.1</td>
<td>Xavier/He initialization scale factor.</td>
</tr>
<tr>
<td><code>seed</code></td>
<td>number</td>
<td>42</td>
<td>Deterministic RNG seed for reproducibility.</td>
</tr>
<tr>
<td><code>verbose</code></td>
<td>boolean</td>
<td>false</td>
<td>Enable debug logging.</td>
</tr>
</table>

---

## 🎯 Optimization Guide

### 📊 By Data Characteristics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PARAMETER SELECTION GUIDE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA TYPE              RECOMMENDED SETTINGS                                │
│  ─────────              ────────────────────                                │
│                                                                             │
│  🔄 Fast-changing       maxSequenceLength: 16-32                            │
│     (high frequency)    nBlocks: 2-3                                        │
│                         learningRate: 0.005-0.01                            │
│                                                                             │
│  📈 Slow trends         maxSequenceLength: 128-256                          │
│     (seasonal)          nBlocks: 5-6                                        │
│                         dilationBase: 2-3                                   │
│                                                                             │
│  🌊 Noisy data          useLayerNorm: true                                  │
│                         dropoutRate: 0.1-0.2                                │
│                         l2Lambda: 0.001                                     │
│                         outlierThreshold: 2.0                               │
│                                                                             │
│  🎯 High precision      hiddenChannels: 64-128                              │
│     needed              useTwoLayerBlock: true                              │
│                         activation: "gelu"                                  │
│                                                                             │
│  ⚡ Limited memory      hiddenChannels: 8-16                                │
│                         nBlocks: 2                                          │
│                         useTwoLayerBlock: false                             │
│                                                                             │
│  🔀 Concept drift       adwinEnabled: true                                  │
│     expected            adwinDelta: 0.001                                   │
│                         learningRate: 0.003                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🎚️ Configuration Presets

#### 🏃 Fast & Light (Edge/IoT)

```typescript
const lightConfig = {
  maxSequenceLength: 16,
  hiddenChannels: 8,
  nBlocks: 2,
  kernelSize: 2,
  useTwoLayerBlock: false,
  useLayerNorm: false,
  learningRate: 0.01,
};
```

#### ⚖️ Balanced (General Purpose)

```typescript
const balancedConfig = {
  maxSequenceLength: 64,
  hiddenChannels: 32,
  nBlocks: 4,
  kernelSize: 3,
  useTwoLayerBlock: true,
  activation: "relu",
  learningRate: 0.001,
};
```

#### 🎯 High Accuracy (Maximum Performance)

```typescript
const accurateConfig = {
  maxSequenceLength: 128,
  hiddenChannels: 64,
  nBlocks: 6,
  kernelSize: 3,
  useTwoLayerBlock: true,
  activation: "gelu",
  useLayerNorm: true,
  dropoutRate: 0.1,
  learningRate: 0.0005,
};
```

#### 🔀 Adaptive (Non-stationary Data)

```typescript
const adaptiveConfig = {
  maxSequenceLength: 32,
  hiddenChannels: 32,
  nBlocks: 3,
  adwinEnabled: true,
  adwinDelta: 0.001,
  learningRate: 0.003,
  outlierThreshold: 2.5,
  residualWindowSize: 50,
};
```

### 📈 Tuning Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HYPERPARAMETER TUNING FLOW                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Start with defaults  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
            ┌───────│   Loss decreasing?    │───────┐
            │ NO    └───────────────────────┘ YES   │
            │                                       │
            ▼                                       ▼
    ┌───────────────┐                   ┌───────────────────┐
    │ ↑ learningRate│                   │ Loss plateaued?   │
    │ or            │                   └─────────┬─────────┘
    │ ↑ hiddenChan  │                       │         │
    └───────────────┘                    NO │         │ YES
                                            ▼         ▼
                                    ┌─────────┐   ┌─────────────┐
                                    │ Continue│   │↓learningRate│
                                    │ training│   │or           │
                                    └─────────┘   │↑ l2Lambda   │
                                                  │↑ nBlocks    │
                                                  └─────────────┘
```

---

## 📈 Use Case Examples

### 🌡️ Time Series Forecasting

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// Weather prediction from multiple sensors
const model = new ConvolutionalRegression({
  maxSequenceLength: 48, // 48 hours of history
  maxFutureSteps: 12, // Predict 12 hours ahead
  hiddenChannels: 32,
  nBlocks: 4,
});

// Features: [temperature, humidity, pressure, wind_speed]
// Targets: [temperature, humidity]
const historicalData = loadWeatherData();

// Train on streaming data
for (const observation of historicalData) {
  const result = model.fitOnline({
    xCoordinates: [observation.features],
    yCoordinates: [observation.targets],
  });

  if (result.driftDetected) {
    console.log("⚠️ Weather pattern shift detected!");
  }
}

// Forecast next 12 hours
const forecast = model.predict(12);
console.log(
  "🌡️ Temperature forecast:",
  forecast.predictions.map((p) => p[0].toFixed(1)),
);
console.log("📊 Confidence:", (forecast.confidence * 100).toFixed(0) + "%");
```

### 📊 Multi-Target Regression

```typescript
// Predict multiple outputs simultaneously
const model = new ConvolutionalRegression({
  maxSequenceLength: 32,
  hiddenChannels: 48,
  useDirectMultiHorizon: true, // Direct prediction for all targets
});

// Input: process parameters [temp, pressure, flow_rate, concentration]
// Output: quality metrics [yield, purity, efficiency]
const processData = loadProcessData();

for (const sample of processData) {
  model.fitOnline({
    xCoordinates: [sample.parameters],
    yCoordinates: [sample.quality],
  });
}

const prediction = model.predict();
console.log("📦 Predicted yield:", prediction.predictions[0][0].toFixed(2));
console.log("💎 Predicted purity:", prediction.predictions[0][1].toFixed(2));
console.log(
  "⚡ Predicted efficiency:",
  prediction.predictions[0][2].toFixed(2),
);
```

### 📈 Financial Prediction with Uncertainty

```typescript
const model = new ConvolutionalRegression({
  maxSequenceLength: 60, // 60 trading days
  maxFutureSteps: 5, // 5-day forecast
  hiddenChannels: 64,
  nBlocks: 5,
  activation: "gelu",
  useLayerNorm: true,
  uncertaintyMultiplier: 1.96, // 95% confidence interval
  outlierThreshold: 2.5, // Handle market anomalies
});

// Features: [open, high, low, close, volume, volatility]
const marketData = loadMarketData();

for (const day of marketData) {
  model.fitOnline({
    xCoordinates: [day.features],
    yCoordinates: [[day.close]], // Predict closing price
  });
}

const forecast = model.predict(5);

console.log("\n📈 5-Day Price Forecast:");
console.log("─".repeat(50));
for (let i = 0; i < 5; i++) {
  const pred = forecast.predictions[i][0];
  const lower = forecast.uncertaintyLower[i][0];
  const upper = forecast.uncertaintyUpper[i][0];
  console.log(
    `Day ${i + 1}: $${pred.toFixed(2)} [$${lower.toFixed(2)} - $${
      upper.toFixed(2)
    }]`,
  );
}
console.log(`\n🎯 Confidence: ${(forecast.confidence * 100).toFixed(0)}%`);
```

### 🔄 Online Learning with Drift Detection

```typescript
const model = new ConvolutionalRegression({
  maxSequenceLength: 32,
  adwinEnabled: true,
  adwinDelta: 0.001, // Sensitive drift detection
  learningRate: 0.002, // Slightly higher for adaptation
});

let totalSamples = 0;
let driftCount = 0;

// Simulate streaming data
const dataStream = createDataStream();

for await (const sample of dataStream) {
  const result = model.fitOnline({
    xCoordinates: [sample.features],
    yCoordinates: [sample.targets],
  });

  totalSamples++;

  if (result.driftDetected) {
    driftCount++;
    console.log(`\n🔀 Drift #${driftCount} detected at sample ${totalSamples}`);
    console.log(`   Current loss: ${result.loss.toFixed(4)}`);
    console.log(`   Avg loss: ${result.metrics.avgLoss.toFixed(4)}`);
  }

  // Log progress every 1000 samples
  if (totalSamples % 1000 === 0) {
    console.log(
      `📊 Samples: ${totalSamples}, MAE: ${result.metrics.mae.toFixed(4)}`,
    );
  }
}
```

### 💾 Save and Load Model

```typescript
// Train model
const model = new ConvolutionalRegression({
  maxSequenceLength: 64,
  hiddenChannels: 32,
});

for (const sample of trainingData) {
  model.fitOnline({
    xCoordinates: [sample.x],
    yCoordinates: [sample.y],
  });
}

// Save model
const serialized = model.save();
await Deno.writeTextFile("model.json", serialized);
console.log("✅ Model saved!");

// Load model later
const loaded = await Deno.readTextFile("model.json");
const restoredModel = new ConvolutionalRegression();
restoredModel.load(loaded);
console.log("✅ Model loaded!");

// Continue training or predict
const prediction = restoredModel.predict();
```

### 📐 Model Inspection

```typescript
const model = new ConvolutionalRegression({
  maxSequenceLength: 32,
  hiddenChannels: 16,
  nBlocks: 3,
});

// After training...
const summary = model.getModelSummary();

console.log("═".repeat(60));
console.log("                    MODEL SUMMARY");
console.log("═".repeat(60));
console.log(summary.architecture);
console.log("─".repeat(60));
console.log(`📊 Total Parameters: ${summary.totalParams.toLocaleString()}`);
console.log(`👁️ Receptive Field: ${summary.receptiveField} timesteps`);
console.log(`💾 Memory Usage: ~${(summary.memoryBytes / 1024).toFixed(1)} KB`);
console.log(`📈 Samples Trained: ${summary.sampleCount.toLocaleString()}`);
console.log("═".repeat(60));

// Get normalization stats
const normStats = model.getNormalizationStats();
console.log("\n📐 Normalization Statistics:");
console.log(`   Warmed up: ${normStats.isWarmedUp}`);
console.log(
  `   Input means: [${
    normStats.inputMeans.map((m) => m.toFixed(3)).join(", ")
  }]`,
);
console.log(
  `   Input stds: [${normStats.inputStds.map((s) => s.toFixed(3)).join(", ")}]`,
);
```

---

## 🧮 Mathematical Foundations

### Adam Optimizer

The Adam optimizer combines momentum with adaptive learning rates:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ADAM UPDATE RULE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   First moment estimate (momentum):                                         │
│   m_t = β₁ · m_{t-1} + (1 - β₁) · g_t                                      │
│                                                                             │
│   Second moment estimate (adaptive):                                        │
│   v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²                                     │
│                                                                             │
│   Bias-corrected estimates:                                                 │
│   m̂_t = m_t / (1 - β₁ᵗ)                                                    │
│   v̂_t = v_t / (1 - β₂ᵗ)                                                    │
│                                                                             │
│   Parameter update:                                                         │
│   θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)                                    │
│                                                                             │
│   Where:                                                                    │
│   • g_t = gradient at time t                                                │
│   • α = learning rate (learningRate)                                        │
│   • β₁ = first moment decay (beta1 = 0.9)                                  │
│   • β₂ = second moment decay (beta2 = 0.999)                               │
│   • ε = numerical stability (epsilon = 1e-8)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Welford Online Statistics

Numerically stable running mean and variance:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WELFORD'S ALGORITHM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For each new value x_n:                                                   │
│                                                                             │
│   count = count + 1                                                         │
│   δ = x_n - mean                                                            │
│   mean = mean + δ / count                                                   │
│   δ₂ = x_n - mean                                                           │
│   M₂ = M₂ + δ · δ₂                                                          │
│                                                                             │
│   variance = M₂ / (count - 1)    [sample variance]                          │
│   std = √(max(variance, ε))                                                 │
│                                                                             │
│   Z-score normalization:                                                    │
│   z = (x - mean) / std                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ADWIN Drift Detection

Adaptive Windowing for distribution change detection:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ADWIN DRIFT DETECTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For window W = W₀ ∪ W₁ (split into two subwindows):                      │
│                                                                             │
│   Hoeffding bound:                                                          │
│   ε = √((1/(2m)) · ln(4/δ))                                                │
│                                                                             │
│   Where m = 1/(1/n₀ + 1/n₁) (harmonic mean of subwindow sizes)             │
│                                                                             │
│   Drift detected when:                                                      │
│   |μ₀ - μ₁| > ε                                                             │
│                                                                             │
│   Where:                                                                    │
│   • μ₀, μ₁ = means of subwindows W₀, W₁                                    │
│   • δ = significance parameter (adwinDelta)                                 │
│   • n₀, n₁ = sizes of subwindows                                           │
│                                                                             │
│   On drift: shrink window by removing oldest buckets                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GELU Activation

Gaussian Error Linear Unit (smoother than ReLU):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GELU APPROXIMATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))            │
│                                                                             │
│   Comparison with ReLU:                                                     │
│                                                                             │
│   ReLU(x) = max(0, x)        │  GELU(x) = x · Φ(x)                         │
│                               │  where Φ is the Gaussian CDF               │
│        │                      │                                             │
│        │    /                 │       /~~                                   │
│        │   /                  │      /                                      │
│   ─────┼──/─────              │  ───/────────                               │
│        │                      │                                             │
│        │                      │                                             │
│                                                                             │
│   • GELU is smooth and differentiable everywhere                            │
│   • Better gradient flow for deep networks                                  │
│   • Slightly more expensive to compute                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💾 Serialization

### Save Model State

```typescript
const model = new ConvolutionalRegression({/* config */});

// Train model...

// Save to string
const serialized = model.save();

// Save to file
await Deno.writeTextFile("model.json", serialized);

// Or send over network
await fetch("/api/save-model", {
  method: "POST",
  body: serialized,
});
```

### Load Model State

```typescript
// Load from file
const json = await Deno.readTextFile("model.json");
const model = new ConvolutionalRegression();
model.load(json);

// Model is ready to use
const prediction = model.predict();
```

### What's Serialized

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERIALIZATION CONTENTS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ Model Configuration (all parameters)                                    │
│  ✅ All Network Weights & Biases                                            │
│  ✅ Adam Optimizer State (m, v, timestep)                                   │
│  ✅ Welford Normalization Statistics                                        │
│  ✅ Input Ring Buffer (sequence history)                                    │
│  ✅ Residual Tracker (uncertainty data)                                     │
│  ✅ ADWIN Detector State (if enabled)                                       │
│  ✅ Sample Count                                                            │
│                                                                             │
│  ❌ Temporary Computation Buffers (recreated on load)                       │
│  ❌ Buffer Pool State (recreated on load)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests
on [GitHub](https://github.com/hviana/multivariate-convolutional-regression).

---

## 📄 License

MIT License © 2025 [Henrique Emanoel Viana](https://github.com/hviana)

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

**Made with ❤️ by [Henrique Emanoel Viana](https://github.com/hviana)**

⭐ Star this repo if you find it useful!

</div>
