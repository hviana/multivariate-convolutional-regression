Model: # 🧠 Multivariate Convolutional Regression

<div align="center">

**A powerful Convolutional Neural Network library for Multivariate Regression
with Incremental Online Learning**

[📦 JSR Package](https://jsr.io/@hviana/multivariate-convolutional-regression) •
[🐙 GitHub](https://github.com/hviana/polynomial-regression) •
[👤 Author: Henrique Emanoel Viana](#-author)

</div>

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
- [📖 API Reference](#-api-reference)
- [🎯 Optimization Guide](#-optimization-guide)
- [💡 Use Case Examples](#-use-case-examples)
- [🔧 Advanced Topics](#-advanced-topics)
- [📄 License](#-license)

---

## ✨ Features

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🌟 CORE CAPABILITIES 🌟                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  🔄 Online Learning      │  📊 Multi-Output       │  🎯 Uncertainty Est.    │
│  Stream data in real-    │  Handle multiple       │  95% confidence         │
│  time, no batching       │  target variables      │  intervals included     │
├──────────────────────────┼────────────────────────┼─────────────────────────┤
│  🛡️ Outlier Detection    │  📈 Drift Detection    │  💾 Full Serialization  │
│  Auto-downweight         │  ADWIN algorithm       │  Save/load complete     │
│  anomalous samples       │  detects changes       │  model state            │
├──────────────────────────┼────────────────────────┼─────────────────────────┤
│  ⚡ Adam Optimizer       │  📉 Learning Schedule  │  🔢 Z-Score Norm        │
│  Adaptive learning       │  Cosine warmup &       │  Welford's online       │
│  with momentum           │  decay strategy        │  statistics             │
└─────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Feature Breakdown

| Feature                            | Description                           | Benefit                             |
| ---------------------------------- | ------------------------------------- | ----------------------------------- |
| 🔄 **Incremental Online Learning** | Process samples one at a time         | Memory efficient, real-time updates |
| 🧮 **Convolutional Architecture**  | 1D convolutions extract patterns      | Captures local dependencies in data |
| 📊 **Multivariate Support**        | Multiple inputs → Multiple outputs    | Complex relationship modeling       |
| ⚡ **Adam Optimizer**              | Adaptive learning rates per parameter | Faster convergence, stable training |
| 📉 **Cosine Warmup Schedule**      | Gradual LR increase then decay        | Prevents early divergence           |
| 🔢 **Welford's Algorithm**         | Online mean/variance computation      | Numerically stable normalization    |
| 🛡️ **Outlier Detection**           | Z-score based anomaly detection       | Robust to noisy data                |
| 📈 **ADWIN Drift Detection**       | Adaptive windowing algorithm          | Handles concept drift               |
| 🎯 **Uncertainty Estimation**      | Confidence intervals on predictions   | Quantified prediction reliability   |
| 🔒 **L2 Regularization**           | Weight decay penalty                  | Prevents overfitting                |

---

## 🚀 Quick Start

### Installation

```typescript
// JSR
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";
```

### Basic Usage

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// 1️⃣ Create model instance
const model = new ConvolutionalRegression();

// 2️⃣ Train with streaming data
const result = model.fitOnline({
  xCoordinates: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  yCoordinates: [[10, 20], [30, 40], [50, 60]],
});

console.log(`📉 Loss: ${result.loss.toFixed(4)}`);
console.log(`✅ Converged: ${result.converged}`);

// 3️⃣ Make predictions
const predictions = model.predict(5);

for (const pred of predictions.predictions) {
  console.log(
    `🎯 Predicted: [${pred.predicted.map((v) => v.toFixed(2)).join(", ")}]`,
  );
  console.log(
    `📊 95% CI: [${pred.lowerBound.map((v) => v.toFixed(2)).join(", ")}] - [${
      pred.upperBound.map((v) => v.toFixed(2)).join(", ")
    }]`,
  );
}
```

---

## 🏗️ Architecture

### Neural Network Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NETWORK ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    INPUT                CONVOLUTIONAL LAYERS                    OUTPUT
  (Features)           (Pattern Extraction)                   (Predictions)

 ┌─────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
 │         │      │   Conv1D + ReLU (×N)    │      │                         │
 │  x₁     │      │  ┌─────┐    ┌─────┐     │      │                         │
 │  x₂     │──────│  │Conv │────│ReLU │     │      │      ┌─────────┐        │
 │  x₃     │      │  │ 1D  │    │     │     │──────│──────│  Dense  │────────│──► ŷ₁, ŷ₂, ...
 │  ...    │      │  └─────┘    └─────┘     │      │      │  Layer  │        │
 │  xₙ     │      │       ×hiddenLayers     │      │      └─────────┘        │
 │         │      │                         │      │                         │
 └─────────┘      └─────────────────────────┘      └─────────────────────────┘
      │                      │                               │
      ▼                      ▼                               ▼
  Z-Score              He Init +                      Linear Output
  Normalize            Same Padding                   + Denormalize
```

### Data Flow Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                                  │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌─────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Raw    │    │   Welford    │    │   Forward    │    │    Loss      │
  │  Data   │───▶│   Z-Score    │───▶│    Pass      │───▶│  Compute     │
  │ (x, y)  │    │   Normalize  │    │              │    │   (MSE)      │
  └─────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                │
       ┌────────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   Outlier    │    │   Backward   │    │    Adam      │    │    ADWIN     │
  │   Check &    │───▶│    Pass      │───▶│   Update     │───▶│    Drift     │
  │  Downweight  │    │  (Gradients) │    │  (Weights)   │    │   Detection  │
  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Mathematical Foundations

#### Conv1D Operation

```
y[c, i] = Σₖ Σⱼ W[c, k, j] · x[k, i + j - pad] + b[c]
```

#### ReLU Activation

```
f(x) = max(0, x)
```

#### Adam Optimizer Update Rules

```
m = β₁·m + (1-β₁)·g           # First moment estimate
v = β₂·v + (1-β₂)·g²          # Second moment estimate
m̂ = m / (1-β₁ᵗ)               # Bias-corrected first moment
v̂ = v / (1-β₂ᵗ)               # Bias-corrected second moment
W = W - η·m̂ / (√v̂ + ε)        # Parameter update
```

#### Learning Rate Schedule

```
Warmup (t ≤ warmupSteps):
    lr = baseLR × (t / warmupSteps)

Cosine Decay (t > warmupSteps):
    progress = (t - warmupSteps) / (totalSteps - warmupSteps)
    lr = baseLR × 0.5 × (1 + cos(π × progress))
```

---

## ⚙️ Configuration Parameters

### Complete Configuration Interface

```typescript
interface ConvolutionalRegressionConfig {
  hiddenLayers?: number; // Default: 2
  convolutionsPerLayer?: number; // Default: 32
  kernelSize?: number; // Default: 3
  learningRate?: number; // Default: 0.001
  warmupSteps?: number; // Default: 100
  totalSteps?: number; // Default: 10000
  beta1?: number; // Default: 0.9
  beta2?: number; // Default: 0.999
  epsilon?: number; // Default: 1e-8
  regularizationStrength?: number; // Default: 1e-4
  convergenceThreshold?: number; // Default: 1e-6
  outlierThreshold?: number; // Default: 3.0
  adwinDelta?: number; // Default: 0.002
}
```

---

### 🔷 Network Architecture Parameters

#### `hiddenLayers`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>2</td></tr>
<tr><td><b>Range</b></td><td>1 - 10</td></tr>
</table>

**Description:** Number of convolutional hidden layers in the network.

```
hiddenLayers = 1:     Input → [Conv+ReLU] → Dense → Output
hiddenLayers = 2:     Input → [Conv+ReLU] → [Conv+ReLU] → Dense → Output
hiddenLayers = 3:     Input → [Conv+ReLU]×3 → Dense → Output
```

**🎯 Optimization Guide:**

| Scenario                    | Recommended Value | Reasoning                       |
| --------------------------- | ----------------- | ------------------------------- |
| Simple linear relationships | 1                 | Minimal complexity needed       |
| Standard regression tasks   | 2                 | Good balance of capacity/speed  |
| Complex nonlinear patterns  | 3-4               | More representation power       |
| Very high-dimensional data  | 4-6               | Hierarchical feature extraction |

```typescript
// Simple data pattern
const simpleModel = new ConvolutionalRegression({ hiddenLayers: 1 });

// Complex multi-scale patterns
const complexModel = new ConvolutionalRegression({ hiddenLayers: 4 });
```

---

#### `convolutionsPerLayer`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>32</td></tr>
<tr><td><b>Range</b></td><td>8 - 256</td></tr>
</table>

**Description:** Number of output channels (filters) per convolutional layer.

```
                     convolutionsPerLayer = 16
                    ┌────────────────────────┐
    1 channel  ────▶│  16 learnable filters  │────▶  16 channels
                    └────────────────────────┘

                     convolutionsPerLayer = 64
                    ┌────────────────────────┐
    1 channel  ────▶│  64 learnable filters  │────▶  64 channels
                    └────────────────────────┘
```

**🎯 Optimization Guide:**

| Data Complexity            | Recommended Value | Total Parameters Impact |
| -------------------------- | ----------------- | ----------------------- |
| Low (< 10 features)        | 8-16              | ~500-2,000              |
| Medium (10-50 features)    | 32                | ~5,000-20,000           |
| High (50-200 features)     | 64-128            | ~50,000-200,000         |
| Very High (> 200 features) | 128-256           | ~200,000+               |

```typescript
// Lightweight model for simple patterns
const lightModel = new ConvolutionalRegression({
  convolutionsPerLayer: 16,
});

// Heavy model for complex patterns
const heavyModel = new ConvolutionalRegression({
  convolutionsPerLayer: 128,
});
```

---

#### `kernelSize`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>3</td></tr>
<tr><td><b>Range</b></td><td>1 - 11 (odd numbers)</td></tr>
</table>

**Description:** Size of the convolutional kernel (receptive field).

```
kernelSize = 3:                kernelSize = 5:                kernelSize = 7:
┌───┬───┬───┐                  ┌───┬───┬───┬───┬───┐          ┌───┬───┬───┬───┬───┬───┬───┐
│ w₁│ w₂│ w₃│                  │w₁ │w₂ │w₃ │w₄ │w₅ │          │w₁ │w₂ │w₃ │w₄ │w₅ │w₆ │w₇ │
└───┴───┴───┘                  └───┴───┴───┴───┴───┘          └───┴───┴───┴───┴───┴───┴───┘
Local patterns                 Medium-range patterns          Long-range patterns
```

**🎯 Optimization Guide:**

| Pattern Type            | Recommended Value | Use Case                      |
| ----------------------- | ----------------- | ----------------------------- |
| Very local dependencies | 1                 | Point-wise transformations    |
| Local patterns          | 3                 | Most regression tasks         |
| Medium-range patterns   | 5                 | Time series with short trends |
| Long-range patterns     | 7-11              | Seasonal or cyclic data       |

```typescript
// Local feature extraction
const localModel = new ConvolutionalRegression({ kernelSize: 3 });

// Capture longer-range dependencies
const wideModel = new ConvolutionalRegression({ kernelSize: 7 });
```

---

### 🔷 Optimizer Parameters

#### `learningRate`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>0.001</td></tr>
<tr><td><b>Range</b></td><td>1e-5 - 0.1</td></tr>
</table>

**Description:** Base learning rate for the Adam optimizer.

```
Learning Rate Effect:
                        
     High LR (0.01)     │    Medium LR (0.001)   │    Low LR (0.0001)
     ⚡ Fast but risky   │    ✅ Balanced          │    🐢 Slow but stable
                        │                        │
         ╱╲             │         ╲              │              
        ╱  ╲            │          ╲             │           ╲
       ╱    ╲           │           ╲            │            ╲
      ╱      ╲──Loss    │            ╲───Loss    │             ╲───Loss
     ╱        ╲         │             ╲          │              ╲
────────────────────    │─────────────────────   │──────────────────────
     May overshoot      │    Converges well      │    Very slow progress
```

**🎯 Optimization Guide:**

| Scenario            | Recommended Value | Notes                         |
| ------------------- | ----------------- | ----------------------------- |
| Initial experiments | 0.001             | Start here for most cases     |
| Large datasets      | 0.0001 - 0.001    | More samples = lower LR       |
| Small datasets      | 0.001 - 0.01      | Fewer samples = can be higher |
| Fine-tuning         | 0.0001 - 0.0005   | After initial training        |
| Unstable training   | 0.0001            | If loss is oscillating        |

```typescript
// Standard training
const standardModel = new ConvolutionalRegression({ learningRate: 0.001 });

// Conservative training for sensitive data
const conservativeModel = new ConvolutionalRegression({ learningRate: 0.0001 });

// Aggressive training for quick experiments
const aggressiveModel = new ConvolutionalRegression({ learningRate: 0.005 });
```

---

#### `warmupSteps`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>100</td></tr>
<tr><td><b>Range</b></td><td>0 - 1000</td></tr>
</table>

**Description:** Number of steps for linear learning rate warmup.

```
Learning Rate Schedule with Warmup:

     lr
      │
  max │                    ╭─────────────────╮
      │                 ╱                     ╲
      │              ╱                          ╲
      │           ╱                               ╲
      │        ╱                                    ╲
      │     ╱                                         ╲
  0   │__╱____________________________________________╲___
      │
      └───────┬────────────────────────────────────────────▶ steps
              │
         warmupSteps
```

**🎯 Optimization Guide:**

| Dataset Size       | Recommended Value | Reasoning                     |
| ------------------ | ----------------- | ----------------------------- |
| < 100 samples      | 10-20             | Quick warmup, limited data    |
| 100-1000 samples   | 50-100            | Standard warmup               |
| 1000-10000 samples | 100-200           | Moderate warmup               |
| > 10000 samples    | 200-500           | Extended warmup for stability |

```typescript
// Quick warmup for small datasets
const quickWarmup = new ConvolutionalRegression({ warmupSteps: 20 });

// Extended warmup for large datasets
const extendedWarmup = new ConvolutionalRegression({ warmupSteps: 300 });
```

---

#### `totalSteps`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>10000</td></tr>
<tr><td><b>Range</b></td><td>1000 - 1000000</td></tr>
</table>

**Description:** Total training steps for cosine decay schedule.

```
Effect of totalSteps on Learning Rate Decay:

totalSteps = 5000          totalSteps = 10000         totalSteps = 50000
   (Fast decay)               (Standard)               (Slow decay)

lr│  ╲                     lr│     ╲                  lr│          ╲
  │   ╲                      │      ╲                   │           ╲
  │    ╲                     │       ╲                  │            ╲
  │     ╲                    │        ╲                 │             ╲
  │      ╲                   │         ╲                │              ╲
  └───────────▶ steps        └────────────▶ steps      └────────────────▶ steps
```

**🎯 Optimization Guide:**

| Training Duration   | Recommended Value | Use Case                 |
| ------------------- | ----------------- | ------------------------ |
| Short experiments   | 1000-5000         | Quick prototyping        |
| Standard training   | 10000             | Most applications        |
| Long training       | 50000-100000      | Complex models           |
| Continuous learning | 100000+           | Streaming data scenarios |

```typescript
// Quick experiment
const quickModel = new ConvolutionalRegression({ totalSteps: 2000 });

// Long training for complex patterns
const longModel = new ConvolutionalRegression({ totalSteps: 50000 });
```

---

#### `beta1` & `beta2`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Defaults</b></td><td>β₁ = 0.9, β₂ = 0.999</td></tr>
<tr><td><b>Range</b></td><td>0.0 - 0.9999</td></tr>
</table>

**Description:** Exponential decay rates for Adam's moment estimates.

```
β₁ (First Moment - Momentum):
    Controls how much past gradients influence current direction
    Higher β₁ → More momentum → Smoother updates

β₂ (Second Moment - RMSprop):
    Controls adaptive learning rate per parameter
    Higher β₂ → More stable → Slower adaptation
```

**🎯 Optimization Guide:**

| Scenario           | β₁   | β₂     | Notes                       |
| ------------------ | ---- | ------ | --------------------------- |
| Standard (default) | 0.9  | 0.999  | Works for most cases        |
| Noisy gradients    | 0.9  | 0.9999 | More gradient smoothing     |
| Sparse features    | 0.95 | 0.999  | Higher momentum             |
| Fast adaptation    | 0.8  | 0.99   | Quicker response to changes |

```typescript
// Standard Adam
const standardAdam = new ConvolutionalRegression({
  beta1: 0.9,
  beta2: 0.999,
});

// More aggressive adaptation
const aggressiveAdam = new ConvolutionalRegression({
  beta1: 0.8,
  beta2: 0.99,
});
```

---

#### `epsilon`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>1e-8</td></tr>
<tr><td><b>Range</b></td><td>1e-10 - 1e-4</td></tr>
</table>

**Description:** Numerical stability constant to prevent division by zero.

```
Adam Update: W -= η·m̂ / (√v̂ + ε)
                              ↑
                     Prevents division by zero
                     when v̂ is very small
```

**🎯 Optimization Guide:**

| Scenario             | Recommended Value | Notes                 |
| -------------------- | ----------------- | --------------------- |
| Standard             | 1e-8              | Default, works well   |
| Mixed precision      | 1e-4              | Prevents underflow    |
| Very small gradients | 1e-10             | More precision needed |

---

### 🔷 Regularization Parameters

#### `regularizationStrength`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>1e-4</td></tr>
<tr><td><b>Range</b></td><td>0 - 0.1</td></tr>
</table>

**Description:** L2 regularization (weight decay) strength.

```
Loss = MSE + (λ/2)·‖W‖²
               ↑
       regularizationStrength

Effect on Weights:
┌────────────────────────────────────────────────────────┐
│  λ = 0 (No reg.)    │  λ = 1e-4 (Light)  │  λ = 1e-2 (Heavy) │
│  Weights can grow   │  Gentle constraint │  Strong shrinkage │
│  unbounded          │  on weight size    │  towards zero     │
└────────────────────────────────────────────────────────┘
```

**🎯 Optimization Guide:**

| Scenario             | Recommended Value | Effect                |
| -------------------- | ----------------- | --------------------- |
| No regularization    | 0                 | Might overfit         |
| Light regularization | 1e-5 - 1e-4       | Subtle weight control |
| Standard             | 1e-4              | Good balance          |
| Heavy regularization | 1e-3 - 1e-2       | Prevents overfitting  |
| Very heavy           | 0.1               | Underfitting risk     |

```typescript
// No regularization (for very clean data)
const noReg = new ConvolutionalRegression({ regularizationStrength: 0 });

// Strong regularization (for noisy data or small datasets)
const strongReg = new ConvolutionalRegression({ regularizationStrength: 0.01 });
```

---

### 🔷 Convergence & Detection Parameters

#### `convergenceThreshold`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>1e-6</td></tr>
<tr><td><b>Range</b></td><td>1e-10 - 1e-3</td></tr>
</table>

**Description:** Gradient L2 norm threshold for declaring convergence.

```
Convergence Check:
    ‖∇L‖₂ < convergenceThreshold  →  converged = true

Gradient Norm over Training:
    │
    │╲
    │ ╲
    │  ╲
    │   ╲__
    │      ╲__
────┼─────────────────────────  ← convergenceThreshold
    │            ╲____
    │                 ╲_______
    └──────────────────────────▶
                                  time
```

**🎯 Optimization Guide:**

| Precision Need | Recommended Value | Training Time           |
| -------------- | ----------------- | ----------------------- |
| Quick training | 1e-4              | Faster, less precise    |
| Standard       | 1e-6              | Balanced                |
| High precision | 1e-8              | Slower, more accurate   |
| Research-grade | 1e-10             | Very slow, very precise |

```typescript
// Quick convergence for prototyping
const quickConverge = new ConvolutionalRegression({
  convergenceThreshold: 1e-4,
});

// Strict convergence for production
const strictConverge = new ConvolutionalRegression({
  convergenceThreshold: 1e-8,
});
```

---

#### `outlierThreshold`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>3.0</td></tr>
<tr><td><b>Range</b></td><td>1.5 - 5.0</td></tr>
</table>

**Description:** Z-score threshold for outlier detection and downweighting.

```
Outlier Detection:
    z = |y - ŷ| / σ
    
    if z > outlierThreshold:
        sample_weight = 0.1  (downweighted)
    else:
        sample_weight = 1.0  (normal)

Distribution of Z-scores:
                    
    │        ╭─────╮        
    │      ╱         ╲      
    │    ╱             ╲    
    │  ╱                 ╲  
    │╱                     ╲
────┼───────────────────────────
   -3σ   -2σ   -1σ    0   +1σ   +2σ   +3σ
    │                              │     │
    └──────────── Normal ──────────┘     │
                                  Outlier Zone
```

**🎯 Optimization Guide:**

| Data Quality    | Recommended Value | Detection Rate             |
| --------------- | ----------------- | -------------------------- |
| Very clean data | 4.0-5.0           | Very few outliers detected |
| Standard data   | 3.0               | ~0.3% flagged as outliers  |
| Noisy data      | 2.5               | ~1.2% flagged              |
| Very noisy      | 2.0               | ~4.5% flagged              |

```typescript
// Sensitive to outliers (clean data expected)
const sensitiveModel = new ConvolutionalRegression({ outlierThreshold: 4.0 });

// Robust to outliers (noisy data)
const robustModel = new ConvolutionalRegression({ outlierThreshold: 2.5 });
```

---

#### `adwinDelta`

<table>
<tr><td><b>Type</b></td><td>number</td></tr>
<tr><td><b>Default</b></td><td>0.002</td></tr>
<tr><td><b>Range</b></td><td>0.0001 - 0.1</td></tr>
</table>

**Description:** ADWIN algorithm confidence parameter for drift detection.

```
ADWIN Drift Detection:

                    Window of recent losses
    ┌─────────────────────────────────────────────────┐
    │  μ₀ (old mean)   │   μ₁ (new mean)             │
    │     = 0.05       │      = 0.15                 │
    └─────────────────────────────────────────────────┘
                       ↑
                    Split point

    Drift detected if: |μ₀ - μ₁| ≥ √((1/2m)·ln(4n/δ))
                                              ↑
                                         adwinDelta
```

**🎯 Optimization Guide:**

| Sensitivity       | Recommended Value | Drift Detection       |
| ----------------- | ----------------- | --------------------- |
| Very sensitive    | 0.0001            | Detects small changes |
| Standard          | 0.002             | Balanced detection    |
| Conservative      | 0.01              | Only major drifts     |
| Very conservative | 0.1               | Very few false alarms |

```typescript
// Sensitive drift detection
const sensitiveDrift = new ConvolutionalRegression({ adwinDelta: 0.0005 });

// Conservative drift detection
const conservativeDrift = new ConvolutionalRegression({ adwinDelta: 0.01 });
```

---

## 📖 API Reference

### Constructor

```typescript
new ConvolutionalRegression(config?: ConvolutionalRegressionConfig)
```

### Methods Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            PUBLIC METHODS                                  │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  📊 Training                                                              │
│  ├── fitOnline(data)           Train incrementally on new samples        │
│  │                                                                        │
│  🔮 Prediction                                                            │
│  ├── predict(steps)            Generate future predictions               │
│  │                                                                        │
│  📈 Inspection                                                            │
│  ├── getModelSummary()         Get architecture & training summary       │
│  ├── getWeights()              Get all weights & optimizer state         │
│  ├── getNormalizationStats()   Get mean/std statistics                   │
│  │                                                                        │
│  💾 Persistence                                                           │
│  ├── save()                    Serialize model to JSON string            │
│  ├── load(json)                Restore model from JSON string            │
│  │                                                                        │
│  🔄 Management                                                            │
│  └── reset()                   Clear all state, reinitialize            │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

### `fitOnline(data)`

Train the model incrementally with new samples.

**Signature:**

```typescript
fitOnline(data: { 
    xCoordinates: number[][], 
    yCoordinates: number[][] 
}): FitResult
```

**Parameters:**

| Parameter      | Type         | Description             |
| -------------- | ------------ | ----------------------- |
| `xCoordinates` | `number[][]` | Array of input vectors  |
| `yCoordinates` | `number[][]` | Array of target vectors |

**Returns: `FitResult`**

```typescript
interface FitResult {
  loss: number; // MSE loss for this batch
  gradientNorm: number; // L2 norm of gradients
  effectiveLearningRate: number; // Current LR after schedule
  isOutlier: boolean; // Any outlier detected?
  converged: boolean; // Gradient < threshold?
  sampleIndex: number; // Total samples processed
  driftDetected: boolean; // ADWIN detected drift?
}
```

**Example:**

```typescript
const model = new ConvolutionalRegression();

// Single batch training
const result = model.fitOnline({
  xCoordinates: [[1, 2, 3], [4, 5, 6]],
  yCoordinates: [[10], [20]],
});

console.log(`Loss: ${result.loss}`);
console.log(`Converged: ${result.converged}`);

// Streaming training
for await (const batch of dataStream) {
  const result = model.fitOnline(batch);

  if (result.driftDetected) {
    console.log("⚠️ Concept drift detected!");
  }

  if (result.converged) {
    console.log("✅ Model has converged");
    break;
  }
}
```

---

### `predict(futureSteps)`

Generate predictions with uncertainty estimates.

**Signature:**

```typescript
predict(futureSteps: number): PredictionResult
```

**Returns: `PredictionResult`**

```typescript
interface PredictionResult {
  predictions: SinglePrediction[]; // Array of predictions
  accuracy: number; // Model accuracy metric
  sampleCount: number; // Training samples seen
  isModelReady: boolean; // Enough training data?
}

interface SinglePrediction {
  predicted: number[]; // Point predictions
  lowerBound: number[]; // 95% CI lower bound
  upperBound: number[]; // 95% CI upper bound
  standardError: number[]; // Standard error per output
}
```

**Example:**

```typescript
const result = model.predict(5);

if (!result.isModelReady) {
  console.log("⚠️ Model needs more training data");
}

console.log(`📊 Model Accuracy: ${(result.accuracy * 100).toFixed(2)}%`);

for (let i = 0; i < result.predictions.length; i++) {
  const pred = result.predictions[i];
  console.log(`Step ${i + 1}:`);
  console.log(`  Predicted: ${pred.predicted}`);
  console.log(`  95% CI: [${pred.lowerBound}, ${pred.upperBound}]`);
  console.log(`  Std Error: ${pred.standardError}`);
}
```

---

### `getModelSummary()`

Get comprehensive model information.

**Returns: `ModelSummary`**

```typescript
interface ModelSummary {
  isInitialized: boolean; // Has model been trained?
  inputDimension: number; // Number of input features
  outputDimension: number; // Number of outputs
  hiddenLayers: number; // Conv layer count
  convolutionsPerLayer: number; // Channels per layer
  kernelSize: number; // Convolution kernel size
  totalParameters: number; // Trainable parameter count
  sampleCount: number; // Training samples seen
  accuracy: number; // Current accuracy
  converged: boolean; // Has converged?
  effectiveLearningRate: number; // Current learning rate
  driftCount: number; // Number of drifts detected
}
```

**Example:**

```typescript
const summary = model.getModelSummary();

console.log(`
╔══════════════════════════════════════════╗
║           MODEL SUMMARY                  ║
╠══════════════════════════════════════════╣
║ Architecture                             ║
║   Input Dimension:    ${
  summary.inputDimension.toString().padStart(6)
}         ║
║   Output Dimension:   ${
  summary.outputDimension.toString().padStart(6)
}         ║
║   Hidden Layers:      ${summary.hiddenLayers.toString().padStart(6)}         ║
║   Convolutions/Layer: ${
  summary.convolutionsPerLayer.toString().padStart(6)
}         ║
║   Kernel Size:        ${summary.kernelSize.toString().padStart(6)}         ║
║   Total Parameters:   ${
  summary.totalParameters.toString().padStart(6)
}         ║
╠══════════════════════════════════════════╣
║ Training Status                          ║
║   Samples Processed:  ${summary.sampleCount.toString().padStart(6)}         ║
║   Accuracy:           ${
  (summary.accuracy * 100).toFixed(2).padStart(5)
}%        ║
║   Converged:          ${summary.converged.toString().padStart(6)}         ║
║   Drift Events:       ${summary.driftCount.toString().padStart(6)}         ║
║   Current LR:         ${
  summary.effectiveLearningRate.toExponential(2).padStart(10)
}   ║
╚══════════════════════════════════════════╝
`);
```

---

### `getWeights()`

Retrieve all model weights and optimizer state.

**Returns: `WeightInfo`**

```typescript
interface WeightInfo {
  kernels: number[][][]; // Layer kernels
  biases: number[][][]; // Layer biases
  firstMoment: number[][][]; // Adam m values
  secondMoment: number[][][]; // Adam v values
  updateCount: number; // Number of Adam updates
}
```

---

### `getNormalizationStats()`

Get input/output normalization statistics.

**Returns: `NormalizationStats`**

```typescript
interface NormalizationStats {
  inputMean: number[]; // Mean of input features
  inputStd: number[]; // Std of input features
  outputMean: number[]; // Mean of outputs
  outputStd: number[]; // Std of outputs
  count: number; // Number of samples
}
```

---

### `save()` & `load()`

Serialize and restore complete model state.

**Example:**

```typescript
// Save model
const savedState = model.save();
localStorage.setItem("myModel", savedState);

// Later: restore model
const newModel = new ConvolutionalRegression();
const savedState = localStorage.getItem("myModel");
if (savedState) {
  newModel.load(savedState);
  console.log("✅ Model restored successfully");
}
```

---

### `reset()`

Clear all model state and return to uninitialized.

```typescript
model.reset();
// Model is now fresh, ready for new training
```

---

## 🎯 Optimization Guide

### Scenario-Based Configuration

#### 📈 Time Series Forecasting

```typescript
const timeSeriesModel = new ConvolutionalRegression({
  hiddenLayers: 3,
  convolutionsPerLayer: 64,
  kernelSize: 5, // Capture temporal patterns
  learningRate: 0.0005,
  warmupSteps: 200,
  totalSteps: 50000,
  regularizationStrength: 1e-4,
  adwinDelta: 0.001, // Sensitive to distribution shifts
});
```

#### 🏭 Industrial Sensor Data (Noisy)

```typescript
const industrialModel = new ConvolutionalRegression({
  hiddenLayers: 2,
  convolutionsPerLayer: 32,
  kernelSize: 3,
  learningRate: 0.001,
  outlierThreshold: 2.5, // More robust to outliers
  regularizationStrength: 1e-3, // Stronger regularization
  adwinDelta: 0.002,
});
```

#### 🚀 Real-time Streaming Data

```typescript
const streamingModel = new ConvolutionalRegression({
  hiddenLayers: 1, // Lightweight
  convolutionsPerLayer: 16, // Fast inference
  kernelSize: 3,
  learningRate: 0.002, // Faster adaptation
  warmupSteps: 50, // Quick warmup
  totalSteps: 100000, // Long-running
  adwinDelta: 0.001, // Quick drift detection
});
```

#### 🔬 High-Precision Scientific Data

```typescript
const scientificModel = new ConvolutionalRegression({
  hiddenLayers: 4,
  convolutionsPerLayer: 128,
  kernelSize: 5,
  learningRate: 0.0001, // Careful learning
  warmupSteps: 500,
  totalSteps: 100000,
  convergenceThreshold: 1e-8, // Strict convergence
  regularizationStrength: 1e-5,
  outlierThreshold: 4.0, // Very clean data expected
});
```

#### 📱 Edge Device (Resource Constrained)

```typescript
const edgeModel = new ConvolutionalRegression({
  hiddenLayers: 1,
  convolutionsPerLayer: 8, // Minimal footprint
  kernelSize: 3,
  learningRate: 0.001,
  warmupSteps: 20,
  totalSteps: 5000,
});
```

---

### Parameter Tuning Flowchart

```
                START
                  │
                  ▼
      ┌───────────────────────┐
      │  Use default config   │
      │  & observe loss curve │
      └───────────────────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │   Loss decreasing?    │
      └───────────────────────┘
            │           │
           YES          NO
            │           │
            ▼           ▼
┌─────────────┐  ┌────────────────┐
│  Continue   │  │ Learning rate  │
│  training   │  │  too high?     │
└─────────────┘  └────────────────┘
                      │      │
                     YES     NO
                      │      │
                      ▼      ▼
           ┌──────────┐  ┌──────────────┐
           │ Decrease │  │ Increase     │
           │ LR by 2x │  │ model size   │
           └──────────┘  └──────────────┘
                      │
                      ▼
      ┌───────────────────────┐
      │   Overfitting?        │
      │ (train↓ but val↑)     │
      └───────────────────────┘
            │           │
           YES          NO
            │           │
            ▼           ▼
┌─────────────┐  ┌────────────────┐
│ ↑ L2 reg    │  │   Check for    │
│ ↓ model     │  │   convergence  │
│   capacity  │  │                │
└─────────────┘  └────────────────┘
```

---

## 💡 Use Case Examples

### Example 1: Multi-Output Regression

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// Predict multiple outputs from multiple inputs
const model = new ConvolutionalRegression({
  hiddenLayers: 2,
  convolutionsPerLayer: 32,
});

// Training data: 3 inputs → 2 outputs
const trainingData = {
  xCoordinates: [
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
  ],
  yCoordinates: [
    [6.0, 2.0], // sum, variance proxy
    [9.0, 2.0],
    [12.0, 2.0],
    [15.0, 2.0],
  ],
};

// Train
for (let epoch = 0; epoch < 100; epoch++) {
  const result = model.fitOnline(trainingData);

  if (epoch % 20 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${result.loss.toFixed(6)}`);
  }
}

// Predict
const predictions = model.predict(3);
console.log("\n📊 Predictions:");
predictions.predictions.forEach((p, i) => {
  console.log(
    `  Step ${i + 1}: [${p.predicted.map((v) => v.toFixed(2)).join(", ")}]`,
  );
});
```

### Example 2: Continuous Learning with Drift Detection

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

const model = new ConvolutionalRegression({
  adwinDelta: 0.001, // Sensitive drift detection
  learningRate: 0.001,
});

// Simulate streaming data with concept drift
async function processDataStream() {
  let phase = 1;

  for (let t = 0; t < 1000; t++) {
    // Generate data with drift at t=500
    const x = [Math.sin(t * 0.1), Math.cos(t * 0.1), t * 0.01];
    const y = phase === 1
      ? [x[0] + x[1]] // Phase 1: simple sum
      : [x[0] * x[1] + x[2]]; // Phase 2: different relationship

    if (t === 500) phase = 2; // Introduce drift

    const result = model.fitOnline({
      xCoordinates: [x],
      yCoordinates: [y],
    });

    if (result.driftDetected) {
      console.log(`🔄 Drift detected at t=${t}!`);
      console.log(`   Loss: ${result.loss.toFixed(4)}`);
    }

    // Periodic status
    if (t % 100 === 0) {
      const summary = model.getModelSummary();
      console.log(
        `t=${t}: Accuracy=${
          (summary.accuracy * 100).toFixed(1)
        }%, Drifts=${summary.driftCount}`,
      );
    }
  }
}

processDataStream();
```

### Example 3: Model Persistence

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

// Training phase
async function train() {
  const model = new ConvolutionalRegression({
    hiddenLayers: 2,
    convolutionsPerLayer: 32,
  });

  // Train on data...
  for (let i = 0; i < 1000; i++) {
    model.fitOnline({
      xCoordinates: [[Math.random(), Math.random()]],
      yCoordinates: [[Math.random()]],
    });
  }

  // Save model
  const serialized = model.save();
  await Deno.writeTextFile("model.json", serialized);
  console.log("✅ Model saved!");

  return model.getModelSummary();
}

// Inference phase
async function inference() {
  const model = new ConvolutionalRegression();

  // Load model
  const serialized = await Deno.readTextFile("model.json");
  model.load(serialized);
  console.log("✅ Model loaded!");

  // Make predictions
  const result = model.predict(5);
  return result.predictions;
}

// Usage
const trainingSummary = await train();
console.log("Training completed:", trainingSummary);

const predictions = await inference();
console.log("Predictions:", predictions);
```

### Example 4: Uncertainty-Aware Predictions

```typescript
import { ConvolutionalRegression } from "jsr:@hviana/multivariate-convolutional-regression";

const model = new ConvolutionalRegression();

// Train model
// ... (training code)

// Get predictions with uncertainty
const result = model.predict(10);

console.log("\n📊 Predictions with Confidence Intervals:\n");
console.log("┌──────┬────────────┬─────────────────────┬────────────┐");
console.log("│ Step │ Prediction │      95% CI         │ Std Error  │");
console.log("├──────┼────────────┼─────────────────────┼────────────┤");

result.predictions.forEach((pred, i) => {
  const prediction = pred.predicted[0].toFixed(3);
  const lower = pred.lowerBound[0].toFixed(3);
  const upper = pred.upperBound[0].toFixed(3);
  const stdErr = pred.standardError[0].toFixed(4);

  console.log(
    `│ ${(i + 1).toString().padStart(4)} │ ${
      prediction.padStart(10)
    } │ [${lower}, ${upper}] │ ${stdErr.padStart(10)} │`,
  );
});

console.log("└──────┴────────────┴─────────────────────┴────────────┘");

// Use uncertainty for decision making
const criticalPredictions = result.predictions.filter(
  (p) => p.standardError[0] > 0.5,
);

if (criticalPredictions.length > 0) {
  console.log("\n⚠️ Warning: Some predictions have high uncertainty!");
}
```

---

## 🔧 Advanced Topics

### Internal Algorithms

#### Welford's Online Statistics

```typescript
// Numerically stable one-pass mean/variance computation
update(x):
    n += 1
    δ = x - μ
    μ += δ / n
    M₂ += δ × (x - μ)
    σ² = M₂ / (n - 1)
```

#### ADWIN Drift Detection

```typescript
// Adaptive windowing for distribution change detection
Drift condition: |μ_old - μ_new| ≥ √((1/2m) × ln(4n/δ))
    where m = harmonic mean of window sizes
```

#### He Weight Initialization

```typescript
// Optimal initialization for ReLU networks
W ~ N(0, √(2/fan_in))
```

### Memory Optimization

The library uses `Float64Array` for all numerical computations and preallocates
buffers to minimize garbage collection pressure:

```typescript
// Internal buffer management
- Activation buffers: Preallocated per layer
- Gradient buffers: Reused across backward passes
- Normalization buffers: Persistent across calls
```

### Performance Tips

1. **Batch Similar Samples**: Group similar samples in `fitOnline()` calls
2. **Monitor Convergence**: Stop training when `converged === true`
3. **Use Appropriate Model Size**: Start small, increase if underfitting
4. **Enable Drift Detection**: Use ADWIN for streaming scenarios

---

## 📊 Comparison with Traditional Methods

| Feature           | This Library | Traditional Polynomial | Deep Learning |
| ----------------- | ------------ | ---------------------- | ------------- |
| Online Learning   | ✅ Native    | ❌ Batch only          | ⚠️ Complex    |
| Memory Efficiency | ✅ Constant  | ❌ O(n)                | ⚠️ Variable   |
| Concept Drift     | ✅ ADWIN     | ❌ None                | ❌ Manual     |
| Uncertainty       | ✅ Built-in  | ⚠️ Limited             | ❌ Separate   |
| Dependencies      | ✅ Zero      | ✅ Zero                | ❌ Many       |
| Setup Complexity  | ✅ Simple    | ✅ Simple              | ❌ Complex    |

---

## 📄 License

**MIT License** © 2025 Henrique Emanoel Viana

---

## 👤 Author

<div align="center">

**Henrique Emanoel Viana**

[🐙 GitHub](https://github.com/hviana) • [📦 JSR](https://jsr.io/@hviana)

</div>

---

<div align="center">

Made with ❤️ for the community

**⭐ Star this repo if you find it useful!**

</div>
