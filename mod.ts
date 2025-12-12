/**
 * ConvolutionalRegression: Convolutional Neural Network for Multivariate Regression
 * with Incremental Online Learning, Adam Optimizer, and Z-score Normalization
 *
 * Architecture: Input(inputDim) → [Conv1D(filters, kernelSize, same) → ReLU]×hiddenLayers → Flatten → Dense(outputDim)
 *
 * @module ConvolutionalRegression
 */

// ============================================================================
// INTERFACES & TYPES
// ============================================================================

/**
 * Configuration options for ConvolutionalRegression
 */
export interface ConvolutionalRegressionConfig {
  /** Number of hidden convolutional layers (1-10, default: 2) */
  hiddenLayers?: number;
  /** Number of convolution filters per layer (1-256, default: 32) */
  convolutionsPerLayer?: number;
  /** Size of convolution kernel (default: 3) */
  kernelSize?: number;
  /** Base learning rate for Adam optimizer (default: 0.001) */
  learningRate?: number;
  /** Number of warmup steps for learning rate schedule (default: 100) */
  warmupSteps?: number;
  /** Total training steps for cosine decay schedule (default: 10000) */
  totalSteps?: number;
  /** Adam β₁ - first moment decay (default: 0.9) */
  beta1?: number;
  /** Adam β₂ - second moment decay (default: 0.999) */
  beta2?: number;
  /** Adam ε for numerical stability (default: 1e-8) */
  epsilon?: number;
  /** L2 regularization strength λ (default: 1e-4) */
  regularizationStrength?: number;
  /** Convergence threshold for loss (default: 1e-6) */
  convergenceThreshold?: number;
  /** Z-score threshold for outlier detection (default: 3.0) */
  outlierThreshold?: number;
  /** ADWIN drift detection delta parameter (default: 0.002) */
  adwinDelta?: number;
}

/**
 * Result of a single online fitting step
 */
export interface FitResult {
  /** Current loss value: L = (1/2n)Σ‖y - ŷ‖² + (λ/2)Σ‖W‖² */
  loss: number;
  /** L2 norm of gradients: ‖∇L‖₂ */
  gradientNorm: number;
  /** Current effective learning rate after warmup/decay */
  effectiveLearningRate: number;
  /** Whether current sample was detected as outlier */
  isOutlier: boolean;
  /** Whether model has converged (loss < threshold) */
  converged: boolean;
  /** Current sample index */
  sampleIndex: number;
  /** Whether concept drift was detected via ADWIN */
  driftDetected: boolean;
}

/**
 * Prediction result containing forecasts and confidence bounds
 */
export interface PredictionResult {
  /** Array of predictions for each future step */
  predictions: SinglePrediction[];
  /** Model accuracy: 1/(1 + L̄) where L̄ is running average loss */
  accuracy: number;
  /** Number of samples the model was trained on */
  sampleCount: number;
  /** Whether the model is ready for predictions */
  isModelReady: boolean;
}

/**
 * Single prediction with confidence bounds
 */
export interface SinglePrediction {
  /** Predicted values for each output dimension */
  predicted: number[];
  /** Lower confidence bound (95%) for each output dimension */
  lowerBound: number[];
  /** Upper confidence bound (95%) for each output dimension */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Model weight information
 */
export interface WeightInfo {
  /** Convolution kernel weights [layer][filter][kernel_element] */
  kernels: number[][][];
  /** Bias terms [layer][filter/neuron] */
  biases: number[][];
  /** First moment estimates for Adam [layer][filter][element] */
  firstMoment: number[][][];
  /** Second moment estimates for Adam [layer][filter][element] */
  secondMoment: number[][][];
  /** Number of parameter updates performed */
  updateCount: number;
}

/**
 * Normalization statistics from Welford's algorithm
 */
export interface NormalizationStats {
  /** Running mean for each input dimension */
  inputMean: number[];
  /** Running standard deviation for each input dimension */
  inputStd: number[];
  /** Running mean for each output dimension */
  outputMean: number[];
  /** Running standard deviation for each output dimension */
  outputStd: number[];
  /** Number of samples processed */
  count: number;
}

/**
 * Model summary information
 */
export interface ModelSummary {
  /** Whether the model has been initialized */
  isInitialized: boolean;
  /** Number of input dimensions */
  inputDimension: number;
  /** Number of output dimensions */
  outputDimension: number;
  /** Number of hidden convolutional layers */
  hiddenLayers: number;
  /** Number of convolution filters per layer */
  convolutionsPerLayer: number;
  /** Convolution kernel size */
  kernelSize: number;
  /** Total number of trainable parameters */
  totalParameters: number;
  /** Number of samples processed */
  sampleCount: number;
  /** Current model accuracy: 1/(1 + L̄) */
  accuracy: number;
  /** Whether the model has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

// ============================================================================
// INTERNAL TYPES
// ============================================================================

interface WelfordState {
  mean: Float64Array;
  m2: Float64Array;
  count: number;
}

interface AdamState {
  m: Float64Array;
  v: Float64Array;
}

interface ADWINBucket {
  total: number;
  variance: number;
  count: number;
}

interface ConvLayer {
  kernels: Float64Array;
  biases: Float64Array;
  kernelGrad: Float64Array;
  biasGrad: Float64Array;
  adamM: Float64Array;
  adamV: Float64Array;
  adamBiasM: Float64Array;
  adamBiasV: Float64Array;
  inputChannels: number;
  outputChannels: number;
}

interface DenseLayer {
  weights: Float64Array;
  biases: Float64Array;
  weightGrad: Float64Array;
  biasGrad: Float64Array;
  adamM: Float64Array;
  adamV: Float64Array;
  adamBiasM: Float64Array;
  adamBiasV: Float64Array;
  inputDim: number;
  outputDim: number;
}

interface SerializedState {
  config: ConvolutionalRegressionConfig;
  model: {
    isInitialized: boolean;
    inputDim: number;
    outputDim: number;
    sampleCount: number;
    updateCount: number;
    totalLoss: number;
    converged: boolean;
    driftCount: number;
  };
  weights: {
    convLayers: Array<{
      kernels: number[];
      biases: number[];
      adamM: number[];
      adamV: number[];
      adamBiasM: number[];
      adamBiasV: number[];
      inputChannels: number;
      outputChannels: number;
    }>;
    denseLayer: {
      weights: number[];
      biases: number[];
      adamM: number[];
      adamV: number[];
      adamBiasM: number[];
      adamBiasV: number[];
      inputDim: number;
      outputDim: number;
    } | null;
  } | null;
  normalization: {
    inputMean: number[];
    inputM2: number[];
    outputMean: number[];
    outputM2: number[];
    count: number;
  } | null;
  adwin: {
    buckets: ADWINBucket[][];
    total: number;
    count: number;
    width: number;
  } | null;
}

// ============================================================================
// BUFFER POOL - Minimize allocations
// ============================================================================

class BufferPool {
  private readonly pools: Map<number, Float64Array[]> = new Map();
  private readonly maxPoolSize: number = 8;

  acquire(size: number): Float64Array {
    const pool = this.pools.get(size);
    if (pool !== undefined && pool.length > 0) {
      const buffer = pool.pop()!;
      buffer.fill(0);
      return buffer;
    }
    return new Float64Array(size);
  }

  release(buffer: Float64Array): void {
    const size = buffer.length;
    let pool = this.pools.get(size);
    if (pool === undefined) {
      pool = [];
      this.pools.set(size, pool);
    }
    if (pool.length < this.maxPoolSize) {
      pool.push(buffer);
    }
  }

  clear(): void {
    this.pools.clear();
  }
}

// ============================================================================
// MAIN CLASS
// ============================================================================

export class ConvolutionalRegression {
  // Configuration (readonly after construction)
  private readonly _hiddenLayers: number;
  private readonly _convolutionsPerLayer: number;
  private readonly _kernelSize: number;
  private readonly _learningRate: number;
  private readonly _warmupSteps: number;
  private readonly _totalSteps: number;
  private readonly _beta1: number;
  private readonly _beta2: number;
  private readonly _epsilon: number;
  private readonly _regularizationStrength: number;
  private readonly _convergenceThreshold: number;
  private readonly _outlierThreshold: number;
  private readonly _adwinDelta: number;

  // Model state
  private _isInitialized: boolean = false;
  private _inputDim: number = 0;
  private _outputDim: number = 0;
  private _spatialDim: number = 0;
  private _flattenedDim: number = 0;

  // Layers
  private _convLayers: ConvLayer[] = [];
  private _denseLayer: DenseLayer | null = null;

  // Normalization (Welford's algorithm)
  private _inputWelford: WelfordState | null = null;
  private _outputWelford: WelfordState | null = null;

  // Training state
  private _sampleCount: number = 0;
  private _updateCount: number = 0;
  private _totalLoss: number = 0;
  private _converged: boolean = false;
  private _driftCount: number = 0;

  // ADWIN state
  private _adwinBuckets: ADWINBucket[][] = [];
  private _adwinTotal: number = 0;
  private _adwinCount: number = 0;
  private _adwinWidth: number = 0;

  // Preallocated buffers
  private readonly _bufferPool: BufferPool = new BufferPool();
  private _activations: Float64Array[] = [];
  private _preActivations: Float64Array[] = [];
  private _gradients: Float64Array[] = [];
  private _inputBuffer: Float64Array | null = null;
  private _outputBuffer: Float64Array | null = null;
  private _targetBuffer: Float64Array | null = null;
  private _lastInput: Float64Array | null = null;

  // Cached computations
  private _cachedBeta1Power: number = 1;
  private _cachedBeta2Power: number = 1;

  /**
   * Creates a new ConvolutionalRegression instance
   *
   * @param config - Configuration options
   * @example
   * ```typescript
   * const model = new ConvolutionalRegression({
   *   hiddenLayers: 2,
   *   convolutionsPerLayer: 32,
   *   learningRate: 0.001
   * });
   * ```
   */
  constructor(config: ConvolutionalRegressionConfig = {}) {
    this._hiddenLayers = Math.max(1, Math.min(10, config.hiddenLayers ?? 2));
    this._convolutionsPerLayer = Math.max(
      1,
      Math.min(256, config.convolutionsPerLayer ?? 32),
    );
    this._kernelSize = Math.max(1, config.kernelSize ?? 3);
    this._learningRate = config.learningRate ?? 0.001;
    this._warmupSteps = Math.max(0, config.warmupSteps ?? 100);
    this._totalSteps = Math.max(1, config.totalSteps ?? 10000);
    this._beta1 = config.beta1 ?? 0.9;
    this._beta2 = config.beta2 ?? 0.999;
    this._epsilon = config.epsilon ?? 1e-8;
    this._regularizationStrength = config.regularizationStrength ?? 1e-4;
    this._convergenceThreshold = config.convergenceThreshold ?? 1e-6;
    this._outlierThreshold = config.outlierThreshold ?? 3.0;
    this._adwinDelta = config.adwinDelta ?? 0.002;
  }

  /**
   * Performs incremental online training with a single batch
   *
   * Algorithm:
   * 1. Auto-detect dimensions and initialize network on first call
   * 2. Update running statistics using Welford's algorithm: δ = x - μ, μ += δ/n, M₂ += δ(x - μ)
   * 3. Normalize inputs: x̃ = (x - μ)/(σ + ε)
   * 4. Forward pass: propagate through conv layers with ReLU, then dense layer
   * 5. Compute loss: L = (1/2n)Σ‖y - ŷ‖² + (λ/2)Σ‖W‖²
   * 6. Backpropagation: ∂L/∂W via chain rule
   * 7. Adam update: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², W -= η(m̂)/(√v̂ + ε)
   * 8. ADWIN drift detection: detect when |μ₀ - μ₁| ≥ εcut
   *
   * @param data - Training data with input and output coordinates
   * @returns FitResult with training metrics
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2, 3], [4, 5, 6]],
   *   yCoordinates: [[7, 8], [9, 10]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    if (!xCoordinates || xCoordinates.length === 0) {
      throw new Error("xCoordinates cannot be empty");
    }
    if (!yCoordinates || yCoordinates.length === 0) {
      throw new Error("yCoordinates cannot be empty");
    }
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        "xCoordinates and yCoordinates must have the same length",
      );
    }

    // Lazy initialization on first call
    if (!this._isInitialized) {
      this._initialize(xCoordinates[0].length, yCoordinates[0].length);
    }

    const batchSize = xCoordinates.length;
    let batchLoss = 0;
    let batchGradNorm = 0;
    let outlierCount = 0;
    let driftDetected = false;

    // Process each sample in the batch
    for (let b = 0; b < batchSize; b++) {
      const x = xCoordinates[b];
      const y = yCoordinates[b];

      // Update Welford statistics for normalization
      this._updateWelford(this._inputWelford!, x);
      this._updateWelford(this._outputWelford!, y);

      // Normalize input: x̃ = (x - μ)/(σ + ε)
      this._normalizeInput(x);

      // Normalize target: ỹ = (y - μ)/(σ + ε)
      this._normalizeTarget(y);

      // Forward pass
      this._forward();

      // Compute loss and residuals
      const loss = this._computeLoss();

      // Outlier detection: r = (y - ŷ)/σ; |r| > threshold → outlier
      const isOutlier = this._detectOutlier();
      const sampleWeight = isOutlier ? 0.1 : 1.0;
      if (isOutlier) outlierCount++;

      // Backward pass with Adam update
      const gradNorm = this._backward(sampleWeight);

      // Accumulate batch statistics
      batchLoss += loss * sampleWeight;
      batchGradNorm += gradNorm * gradNorm;

      // ADWIN drift detection
      if (this._detectDrift(loss)) {
        driftDetected = true;
        this._handleDrift();
      }

      this._sampleCount++;

      // Store last input for prediction
      this._copyToLastInput(x);
    }

    // Update total loss for accuracy calculation: L̄ = ΣLoss/n
    this._totalLoss += batchLoss;

    const avgLoss = batchLoss / batchSize;
    this._converged = avgLoss < this._convergenceThreshold;

    return {
      loss: avgLoss,
      gradientNorm: Math.sqrt(batchGradNorm / batchSize),
      effectiveLearningRate: this._getEffectiveLR(),
      isOutlier: outlierCount > batchSize / 2,
      converged: this._converged,
      sampleIndex: this._sampleCount,
      driftDetected,
    };
  }

  /**
   * Initialize network architecture
   *
   * Structure: Input(inputDim) → [Conv1D(filters, kernelSize, same) → ReLU]×hiddenLayers → Flatten → Dense(outputDim)
   *
   * Weight initialization: He initialization W ~ N(0, √(2/fan_in))
   */
  private _initialize(inputDim: number, outputDim: number): void {
    this._inputDim = inputDim;
    this._outputDim = outputDim;
    this._spatialDim = inputDim;

    // Initialize Welford states
    this._inputWelford = {
      mean: new Float64Array(inputDim),
      m2: new Float64Array(inputDim),
      count: 0,
    };
    this._outputWelford = {
      mean: new Float64Array(outputDim),
      m2: new Float64Array(outputDim),
      count: 0,
    };

    // Create conv layers
    this._convLayers = [];
    let inChannels = 1;

    for (let l = 0; l < this._hiddenLayers; l++) {
      const outChannels = this._convolutionsPerLayer;
      const kernelElements = inChannels * this._kernelSize * outChannels;

      // He initialization: std = √(2 / (inChannels * kernelSize))
      const std = Math.sqrt(2.0 / (inChannels * this._kernelSize));

      const kernels = new Float64Array(kernelElements);
      for (let i = 0; i < kernelElements; i++) {
        kernels[i] = this._randomNormal() * std;
      }

      this._convLayers.push({
        kernels,
        biases: new Float64Array(outChannels),
        kernelGrad: new Float64Array(kernelElements),
        biasGrad: new Float64Array(outChannels),
        adamM: new Float64Array(kernelElements),
        adamV: new Float64Array(kernelElements),
        adamBiasM: new Float64Array(outChannels),
        adamBiasV: new Float64Array(outChannels),
        inputChannels: inChannels,
        outputChannels: outChannels,
      });

      inChannels = outChannels;
    }

    // Flattened dimension after conv layers
    this._flattenedDim = this._convolutionsPerLayer * this._spatialDim;

    // Create dense layer
    const denseWeights = this._flattenedDim * outputDim;
    const denseStd = Math.sqrt(2.0 / this._flattenedDim);

    const weights = new Float64Array(denseWeights);
    for (let i = 0; i < denseWeights; i++) {
      weights[i] = this._randomNormal() * denseStd;
    }

    this._denseLayer = {
      weights,
      biases: new Float64Array(outputDim),
      weightGrad: new Float64Array(denseWeights),
      biasGrad: new Float64Array(outputDim),
      adamM: new Float64Array(denseWeights),
      adamV: new Float64Array(denseWeights),
      adamBiasM: new Float64Array(outputDim),
      adamBiasV: new Float64Array(outputDim),
      inputDim: this._flattenedDim,
      outputDim,
    };

    // Preallocate activation buffers
    this._activations = [];
    this._preActivations = [];
    this._gradients = [];

    // Input activation (1 channel × spatialDim)
    this._activations.push(new Float64Array(this._spatialDim));
    this._preActivations.push(new Float64Array(this._spatialDim));

    // Conv layer activations
    for (let l = 0; l < this._hiddenLayers; l++) {
      const size = this._convolutionsPerLayer * this._spatialDim;
      this._activations.push(new Float64Array(size));
      this._preActivations.push(new Float64Array(size));
      this._gradients.push(new Float64Array(size));
    }

    // Dense layer output
    this._activations.push(new Float64Array(outputDim));
    this._gradients.push(new Float64Array(this._flattenedDim));
    this._gradients.push(new Float64Array(outputDim));

    // Working buffers
    this._inputBuffer = new Float64Array(inputDim);
    this._outputBuffer = new Float64Array(outputDim);
    this._targetBuffer = new Float64Array(outputDim);
    this._lastInput = new Float64Array(inputDim);

    // Initialize ADWIN
    this._adwinBuckets = [];
    this._adwinTotal = 0;
    this._adwinCount = 0;
    this._adwinWidth = 0;

    this._isInitialized = true;
  }

  /**
   * Box-Muller transform for normal random numbers
   * Returns sample from N(0, 1)
   */
  private _randomNormal(): number {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Update Welford running statistics
   *
   * Welford's algorithm for numerical stability:
   * δ = x - μ
   * μ += δ/n
   * M₂ += δ(x - μ)
   * σ² = M₂/(n-1)
   */
  private _updateWelford(state: WelfordState, values: number[]): void {
    state.count++;
    const n = state.count;
    const mean = state.mean;
    const m2 = state.m2;

    for (let i = 0, len = values.length; i < len; i++) {
      const x = values[i];
      const delta = x - mean[i];
      mean[i] += delta / n;
      const delta2 = x - mean[i];
      m2[i] += delta * delta2;
    }
  }

  /**
   * Normalize input using z-score: x̃ = (x - μ)/(σ + ε)
   */
  private _normalizeInput(input: number[]): void {
    const state = this._inputWelford!;
    const n = state.count;
    const activation = this._activations[0];

    for (let i = 0, len = this._inputDim; i < len; i++) {
      const variance = n > 1 ? state.m2[i] / (n - 1) : 1;
      const std = Math.sqrt(variance + this._epsilon);
      activation[i] = (input[i] - state.mean[i]) / std;
    }
  }

  /**
   * Normalize target output
   */
  private _normalizeTarget(output: number[]): void {
    const state = this._outputWelford!;
    const n = state.count;
    const target = this._targetBuffer!;

    for (let i = 0, len = this._outputDim; i < len; i++) {
      const variance = n > 1 ? state.m2[i] / (n - 1) : 1;
      const std = Math.sqrt(variance + this._epsilon);
      target[i] = (output[i] - state.mean[i]) / std;
    }
  }

  /**
   * Forward pass through the network
   *
   * Conv1D with same padding: y[c,i] = Σₖ Σⱼ(W[c,k,j] · x[k,i+j-pad]) + b[c]
   * ReLU activation: a = max(0, z)
   */
  private _forward(): void {
    const pad = Math.floor(this._kernelSize / 2);
    const spatialDim = this._spatialDim;
    const kernelSize = this._kernelSize;

    // Forward through conv layers
    for (let l = 0; l < this._hiddenLayers; l++) {
      const layer = this._convLayers[l];
      const input = this._activations[l];
      const preAct = this._preActivations[l + 1];
      const output = this._activations[l + 1];
      const { kernels, biases, inputChannels, outputChannels } = layer;

      // Clear pre-activation buffer
      preAct.fill(0);

      // Conv1D: y[oc,i] = Σ(ic) Σ(k) W[oc,ic,k] · x[ic, i+k-pad] + b[oc]
      for (let oc = 0; oc < outputChannels; oc++) {
        const bias = biases[oc];

        for (let i = 0; i < spatialDim; i++) {
          let sum = bias;

          for (let ic = 0; ic < inputChannels; ic++) {
            const kernelBase = (oc * inputChannels + ic) * kernelSize;
            const inputBase = ic * spatialDim;

            for (let k = 0; k < kernelSize; k++) {
              const inputIdx = i + k - pad;
              // Same padding: clamp to valid range
              const clampedIdx = inputIdx < 0
                ? 0
                : (inputIdx >= spatialDim ? spatialDim - 1 : inputIdx);
              sum += kernels[kernelBase + k] * input[inputBase + clampedIdx];
            }
          }

          const outIdx = oc * spatialDim + i;
          preAct[outIdx] = sum;
          // ReLU: a = max(0, z)
          output[outIdx] = sum > 0 ? sum : 0;
        }
      }
    }

    // Dense layer forward: y = Wx + b
    const flattenedInput = this._activations[this._hiddenLayers];
    const denseOutput = this._activations[this._hiddenLayers + 1];
    const dense = this._denseLayer!;

    for (let o = 0; o < dense.outputDim; o++) {
      let sum = dense.biases[o];
      const weightBase = o * dense.inputDim;

      for (let i = 0; i < dense.inputDim; i++) {
        sum += dense.weights[weightBase + i] * flattenedInput[i];
      }

      denseOutput[o] = sum;
    }
  }

  /**
   * Compute MSE loss with L2 regularization
   *
   * L = (1/2n)Σ‖y - ŷ‖² + (λ/2)Σ‖W‖²
   */
  private _computeLoss(): number {
    const output = this._activations[this._hiddenLayers + 1];
    const target = this._targetBuffer!;
    const outputGrad = this._gradients[this._gradients.length - 1];

    let mse = 0;
    const n = this._outputDim;

    // MSE component and output gradient
    for (let i = 0; i < n; i++) {
      const diff = output[i] - target[i];
      outputGrad[i] = diff / n;
      mse += diff * diff;
    }
    mse /= 2 * n;

    // L2 regularization: (λ/2)Σ‖W‖²
    let l2 = 0;

    for (let l = 0; l < this._hiddenLayers; l++) {
      const kernels = this._convLayers[l].kernels;
      for (let i = 0, len = kernels.length; i < len; i++) {
        l2 += kernels[i] * kernels[i];
      }
    }

    const denseWeights = this._denseLayer!.weights;
    for (let i = 0, len = denseWeights.length; i < len; i++) {
      l2 += denseWeights[i] * denseWeights[i];
    }

    return mse + (this._regularizationStrength / 2) * l2;
  }

  /**
   * Detect outlier based on standardized residuals
   * r = (y - ŷ)/σ; |r| > threshold → outlier
   */
  private _detectOutlier(): boolean {
    const output = this._activations[this._hiddenLayers + 1];
    const target = this._targetBuffer!;
    const state = this._outputWelford!;
    const n = state.count;

    if (n < 2) return false;

    for (let i = 0; i < this._outputDim; i++) {
      const variance = state.m2[i] / (n - 1);
      const std = Math.sqrt(variance + this._epsilon);
      const residual = Math.abs(output[i] - target[i]) / std;

      if (residual > this._outlierThreshold) {
        return true;
      }
    }

    return false;
  }

  /**
   * Backward pass with Adam optimizer
   *
   * Gradient computation:
   * ∂L/∂W[conv] = upstream_grad * local_grad (via chain rule)
   * ∂Conv/∂W: gradient w.r.t kernel weights
   *
   * Adam update:
   * m = β₁m + (1-β₁)g
   * v = β₂v + (1-β₂)g²
   * m̂ = m/(1-β₁ᵗ)
   * v̂ = v/(1-β₂ᵗ)
   * W -= η · m̂/(√v̂ + ε)
   */
  private _backward(sampleWeight: number): number {
    this._updateCount++;

    // Update cached powers for bias correction
    this._cachedBeta1Power *= this._beta1;
    this._cachedBeta2Power *= this._beta2;

    const lr = this._getEffectiveLR();
    const beta1 = this._beta1;
    const beta2 = this._beta2;
    const eps = this._epsilon;
    const lambda = this._regularizationStrength;
    const bias1Correction = 1 - this._cachedBeta1Power;
    const bias2Correction = 1 - this._cachedBeta2Power;

    let gradNormSq = 0;

    // Dense layer backward
    const denseInput = this._activations[this._hiddenLayers];
    const denseOutputGrad = this._gradients[this._gradients.length - 1];
    const denseInputGrad = this._gradients[this._gradients.length - 2];
    const dense = this._denseLayer!;

    // Compute input gradient for dense layer
    denseInputGrad.fill(0);
    for (let o = 0; o < dense.outputDim; o++) {
      const grad = denseOutputGrad[o] * sampleWeight;
      const weightBase = o * dense.inputDim;

      for (let i = 0; i < dense.inputDim; i++) {
        denseInputGrad[i] += dense.weights[weightBase + i] * grad;
      }
    }

    // Update dense weights with Adam
    for (let o = 0; o < dense.outputDim; o++) {
      const grad = denseOutputGrad[o] * sampleWeight;
      const weightBase = o * dense.inputDim;

      for (let i = 0; i < dense.inputDim; i++) {
        const idx = weightBase + i;
        const g = grad * denseInput[i] + lambda * dense.weights[idx];
        gradNormSq += g * g;

        // Adam: m = β₁m + (1-β₁)g
        dense.adamM[idx] = beta1 * dense.adamM[idx] + (1 - beta1) * g;
        // Adam: v = β₂v + (1-β₂)g²
        dense.adamV[idx] = beta2 * dense.adamV[idx] + (1 - beta2) * g * g;

        // Bias-corrected estimates and update
        const mHat = dense.adamM[idx] / bias1Correction;
        const vHat = dense.adamV[idx] / bias2Correction;
        dense.weights[idx] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }

    // Update dense biases with Adam
    for (let o = 0; o < dense.outputDim; o++) {
      const g = denseOutputGrad[o] * sampleWeight;
      gradNormSq += g * g;

      dense.adamBiasM[o] = beta1 * dense.adamBiasM[o] + (1 - beta1) * g;
      dense.adamBiasV[o] = beta2 * dense.adamBiasV[o] + (1 - beta2) * g * g;

      const mHat = dense.adamBiasM[o] / bias1Correction;
      const vHat = dense.adamBiasV[o] / bias2Correction;
      dense.biases[o] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }

    // Conv layers backward (reverse order)
    const pad = Math.floor(this._kernelSize / 2);
    const spatialDim = this._spatialDim;
    const kernelSize = this._kernelSize;

    let upstream = denseInputGrad;

    for (let l = this._hiddenLayers - 1; l >= 0; l--) {
      const layer = this._convLayers[l];
      const {
        kernels,
        biases,
        adamM,
        adamV,
        adamBiasM,
        adamBiasV,
        inputChannels,
        outputChannels,
      } = layer;
      const input = this._activations[l];
      const preAct = this._preActivations[l + 1];
      const downstream = l > 0 ? this._gradients[l - 1] : null;

      // Apply ReLU derivative to upstream gradient
      // ReLU'(x) = 1 if x > 0, else 0
      for (let i = 0, len = upstream.length; i < len; i++) {
        upstream[i] = preAct[i] > 0 ? upstream[i] * sampleWeight : 0;
      }

      // Initialize downstream gradient if needed
      if (downstream) {
        downstream.fill(0);
      }

      // Compute gradients and update
      for (let oc = 0; oc < outputChannels; oc++) {
        // Bias gradient
        let biasGrad = 0;
        for (let i = 0; i < spatialDim; i++) {
          biasGrad += upstream[oc * spatialDim + i];
        }
        gradNormSq += biasGrad * biasGrad;

        // Update bias with Adam
        adamBiasM[oc] = beta1 * adamBiasM[oc] + (1 - beta1) * biasGrad;
        adamBiasV[oc] = beta2 * adamBiasV[oc] +
          (1 - beta2) * biasGrad * biasGrad;
        const bMHat = adamBiasM[oc] / bias1Correction;
        const bVHat = adamBiasV[oc] / bias2Correction;
        biases[oc] -= lr * bMHat / (Math.sqrt(bVHat) + eps);

        // Kernel gradients and input gradients
        for (let ic = 0; ic < inputChannels; ic++) {
          const kernelBase = (oc * inputChannels + ic) * kernelSize;
          const inputBase = ic * spatialDim;

          for (let k = 0; k < kernelSize; k++) {
            let kernelGrad = lambda * kernels[kernelBase + k];

            for (let i = 0; i < spatialDim; i++) {
              const inputIdx = i + k - pad;
              const clampedIdx = inputIdx < 0
                ? 0
                : (inputIdx >= spatialDim ? spatialDim - 1 : inputIdx);
              const upstreamVal = upstream[oc * spatialDim + i];

              kernelGrad += upstreamVal * input[inputBase + clampedIdx];

              // Accumulate input gradient
              if (downstream && inputIdx >= 0 && inputIdx < spatialDim) {
                downstream[inputBase + inputIdx] += kernels[kernelBase + k] *
                  upstreamVal;
              }
            }

            gradNormSq += kernelGrad * kernelGrad;

            // Update kernel with Adam
            const idx = kernelBase + k;
            adamM[idx] = beta1 * adamM[idx] + (1 - beta1) * kernelGrad;
            adamV[idx] = beta2 * adamV[idx] +
              (1 - beta2) * kernelGrad * kernelGrad;
            const mHat = adamM[idx] / bias1Correction;
            const vHat = adamV[idx] / bias2Correction;
            kernels[idx] -= lr * mHat / (Math.sqrt(vHat) + eps);
          }
        }
      }

      if (downstream) {
        upstream = downstream;
      }
    }

    return Math.sqrt(gradNormSq);
  }

  /**
   * Get effective learning rate with warmup and cosine decay
   *
   * Warmup (t < warmupSteps): lr = baseLR × (t / warmupSteps)
   * Cosine decay: lr = baseLR × 0.5 × (1 + cos(π × progress))
   * where progress = (t - warmupSteps) / (totalSteps - warmupSteps)
   */
  private _getEffectiveLR(): number {
    const t = this._updateCount;

    if (t < this._warmupSteps) {
      // Linear warmup
      return this._learningRate * (t + 1) / this._warmupSteps;
    }

    // Cosine decay
    const progress = (t - this._warmupSteps) /
      Math.max(1, this._totalSteps - this._warmupSteps);
    const clampedProgress = Math.min(1, progress);
    return this._learningRate * 0.5 * (1 + Math.cos(Math.PI * clampedProgress));
  }

  /**
   * ADWIN drift detection
   *
   * Adaptive windowing: maintains error history, detects drift when
   * |μ₀ - μ₁| ≥ εcut where εcut = √((2/m)·ln(2/δ))
   */
  private _detectDrift(loss: number): boolean {
    // Add to window
    this._adwinTotal += loss;
    this._adwinCount++;
    this._adwinWidth++;

    // Add new bucket at level 0
    if (this._adwinBuckets.length === 0) {
      this._adwinBuckets.push([]);
    }
    this._adwinBuckets[0].push({ total: loss, variance: 0, count: 1 });

    // Compress buckets (exponential histogram)
    this._compressBuckets();

    // Check for drift
    if (this._adwinWidth < 10) return false;

    return this._checkDriftCondition();
  }

  /**
   * Compress ADWIN buckets using exponential histogram
   */
  private _compressBuckets(): void {
    const maxBuckets = 2;

    for (let level = 0; level < this._adwinBuckets.length; level++) {
      while (this._adwinBuckets[level].length > maxBuckets) {
        const b1 = this._adwinBuckets[level].shift()!;
        const b2 = this._adwinBuckets[level].shift()!;

        const merged: ADWINBucket = {
          total: b1.total + b2.total,
          variance: 0,
          count: b1.count + b2.count,
        };

        if (level + 1 >= this._adwinBuckets.length) {
          this._adwinBuckets.push([]);
        }
        this._adwinBuckets[level + 1].push(merged);
      }
    }
  }

  /**
   * Check ADWIN drift condition: |μ₀ - μ₁| ≥ εcut
   */
  private _checkDriftCondition(): boolean {
    const totalN = this._adwinWidth;
    if (totalN < 2) return false;

    let n0 = 0;
    let sum0 = 0;

    // Iterate through buckets from oldest to newest
    for (let level = this._adwinBuckets.length - 1; level >= 0; level--) {
      const buckets = this._adwinBuckets[level];
      for (let i = 0; i < buckets.length; i++) {
        const bucket = buckets[i];
        n0 += bucket.count;
        sum0 += bucket.total;

        const n1 = totalN - n0;
        const sum1 = this._adwinTotal - sum0;

        if (n0 > 0 && n1 > 0) {
          const mean0 = sum0 / n0;
          const mean1 = sum1 / n1;
          const m = 1.0 / (1.0 / n0 + 1.0 / n1);
          const epsCut = Math.sqrt(2.0 / m * Math.log(2.0 / this._adwinDelta));

          if (Math.abs(mean0 - mean1) >= epsCut) {
            return true;
          }
        }
      }
    }

    return false;
  }

  /**
   * Handle detected drift by shrinking window and partially resetting stats
   */
  private _handleDrift(): void {
    this._driftCount++;

    // Remove oldest buckets until drift condition no longer holds
    while (this._adwinBuckets.length > 0) {
      const lastLevel = this._adwinBuckets.length - 1;
      if (this._adwinBuckets[lastLevel].length === 0) {
        this._adwinBuckets.pop();
        continue;
      }

      const oldBucket = this._adwinBuckets[lastLevel].pop()!;
      this._adwinTotal -= oldBucket.total;
      this._adwinCount -= oldBucket.count;
      this._adwinWidth -= oldBucket.count;

      if (!this._checkDriftCondition()) {
        break;
      }
    }

    // Decay normalization statistics
    const decayFactor = 0.5;
    if (this._inputWelford) {
      const m2 = this._inputWelford.m2;
      for (let i = 0, len = m2.length; i < len; i++) {
        m2[i] *= decayFactor;
      }
    }
    if (this._outputWelford) {
      const m2 = this._outputWelford.m2;
      for (let i = 0, len = m2.length; i < len; i++) {
        m2[i] *= decayFactor;
      }
    }
  }

  /**
   * Copy input to lastInput buffer for predictions
   */
  private _copyToLastInput(input: number[]): void {
    const last = this._lastInput!;
    for (let i = 0, len = input.length; i < len; i++) {
      last[i] = input[i];
    }
  }

  /**
   * Generate predictions for future steps
   *
   * Uses autoregressive prediction: output at step t becomes input for step t+1
   * Confidence bounds computed from output variance with increasing uncertainty
   *
   * @param futureSteps - Number of future steps to predict
   * @returns PredictionResult with predictions and confidence bounds
   * @example
   * ```typescript
   * const result = model.predict(5);
   * for (const pred of result.predictions) {
   *   console.log(`Predicted: ${pred.predicted}, Bounds: [${pred.lowerBound}, ${pred.upperBound}]`);
   * }
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    if (!this._isInitialized || this._sampleCount < 1) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this._sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const accuracy = 1 / (1 + this._totalLoss / this._sampleCount);

    // Get output standard deviations for confidence bounds
    const outputState = this._outputWelford!;
    const n = outputState.count;
    const outputStds: number[] = new Array(this._outputDim);

    for (let i = 0; i < this._outputDim; i++) {
      const variance = n > 1 ? outputState.m2[i] / (n - 1) : 1;
      outputStds[i] = Math.sqrt(variance + this._epsilon);
    }

    // Use last input as starting point
    const currentInput = this._bufferPool.acquire(this._inputDim);
    currentInput.set(this._lastInput!);

    for (let step = 0; step < futureSteps; step++) {
      // Normalize input
      this._normalizeInputBuffer(currentInput);

      // Forward pass
      this._forward();

      // Get normalized output
      const normalizedOutput = this._activations[this._hiddenLayers + 1];

      // Denormalize to original scale
      const predicted: number[] = new Array(this._outputDim);
      const lowerBound: number[] = new Array(this._outputDim);
      const upperBound: number[] = new Array(this._outputDim);
      const standardError: number[] = new Array(this._outputDim);

      // 95% confidence interval with increasing uncertainty
      const confidenceMultiplier = 1.96 * Math.sqrt(1 + step * 0.1);

      for (let i = 0; i < this._outputDim; i++) {
        const variance = n > 1 ? outputState.m2[i] / (n - 1) : 1;
        const std = Math.sqrt(variance + this._epsilon);

        predicted[i] = normalizedOutput[i] * std + outputState.mean[i];

        const bound = confidenceMultiplier * outputStds[i];
        lowerBound[i] = predicted[i] - bound;
        upperBound[i] = predicted[i] + bound;
        standardError[i] = outputStds[i] / Math.sqrt(Math.max(1, n)) *
          Math.sqrt(1 + step * 0.1);
      }

      predictions.push({ predicted, lowerBound, upperBound, standardError });

      // Autoregressive: shift input and append prediction
      if (this._outputDim <= this._inputDim) {
        for (let i = 0; i < this._inputDim - this._outputDim; i++) {
          currentInput[i] = currentInput[i + this._outputDim];
        }
        for (let i = 0; i < this._outputDim; i++) {
          currentInput[this._inputDim - this._outputDim + i] = predicted[i];
        }
      } else {
        for (let i = 0; i < this._inputDim; i++) {
          currentInput[i] = predicted[i % this._outputDim];
        }
      }
    }

    this._bufferPool.release(currentInput);

    return {
      predictions,
      accuracy,
      sampleCount: this._sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Normalize input buffer in-place
   */
  private _normalizeInputBuffer(input: Float64Array): void {
    const state = this._inputWelford!;
    const n = state.count;
    const activation = this._activations[0];

    for (let i = 0; i < this._inputDim; i++) {
      const variance = n > 1 ? state.m2[i] / (n - 1) : 1;
      const std = Math.sqrt(variance + this._epsilon);
      activation[i] = (input[i] - state.mean[i]) / std;
    }
  }

  /**
   * Get comprehensive model summary
   *
   * @returns ModelSummary with architecture and training details
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Parameters: ${summary.totalParameters}, Accuracy: ${summary.accuracy}`);
   * ```
   */
  getModelSummary(): ModelSummary {
    let totalParameters = 0;

    if (this._isInitialized) {
      // Conv layer parameters
      for (const layer of this._convLayers) {
        totalParameters += layer.kernels.length + layer.biases.length;
      }
      // Dense layer parameters
      if (this._denseLayer) {
        totalParameters += this._denseLayer.weights.length +
          this._denseLayer.biases.length;
      }
    }

    const avgLoss = this._sampleCount > 0
      ? this._totalLoss / this._sampleCount
      : 0;

    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inputDim,
      outputDimension: this._outputDim,
      hiddenLayers: this._hiddenLayers,
      convolutionsPerLayer: this._convolutionsPerLayer,
      kernelSize: this._kernelSize,
      totalParameters,
      sampleCount: this._sampleCount,
      accuracy: 1 / (1 + avgLoss),
      converged: this._converged,
      effectiveLearningRate: this._getEffectiveLR(),
      driftCount: this._driftCount,
    };
  }

  /**
   * Get current model weights and Adam optimizer states
   *
   * @returns WeightInfo containing all trainable parameters
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * console.log(`Layers: ${weights.kernels.length}, Updates: ${weights.updateCount}`);
   * ```
   */
  getWeights(): WeightInfo {
    const kernels: number[][][] = [];
    const biases: number[][] = [];
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];

    if (this._isInitialized) {
      // Conv layers
      for (const layer of this._convLayers) {
        const layerKernels: number[][] = [];
        const layerM: number[][] = [];
        const layerV: number[][] = [];

        const kernelSize = layer.kernels.length / layer.outputChannels;
        for (let oc = 0; oc < layer.outputChannels; oc++) {
          const start = oc * kernelSize;
          layerKernels.push(
            Array.from(layer.kernels.subarray(start, start + kernelSize)),
          );
          layerM.push(
            Array.from(layer.adamM.subarray(start, start + kernelSize)),
          );
          layerV.push(
            Array.from(layer.adamV.subarray(start, start + kernelSize)),
          );
        }

        kernels.push(layerKernels);
        biases.push(Array.from(layer.biases));
        firstMoment.push(layerM);
        secondMoment.push(layerV);
      }

      // Dense layer
      if (this._denseLayer) {
        kernels.push([Array.from(this._denseLayer.weights)]);
        biases.push(Array.from(this._denseLayer.biases));
        firstMoment.push([Array.from(this._denseLayer.adamM)]);
        secondMoment.push([Array.from(this._denseLayer.adamV)]);
      }
    }

    return {
      kernels,
      biases,
      firstMoment,
      secondMoment,
      updateCount: this._updateCount,
    };
  }

  /**
   * Get normalization statistics from Welford's algorithm
   *
   * @returns NormalizationStats with running mean and std for inputs/outputs
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   * console.log(`Input mean: [${stats.inputMean.join(', ')}]`);
   * ```
   */
  getNormalizationStats(): NormalizationStats {
    const inputMean: number[] = [];
    const inputStd: number[] = [];
    const outputMean: number[] = [];
    const outputStd: number[] = [];
    let count = 0;

    if (this._inputWelford && this._outputWelford) {
      count = this._inputWelford.count;
      const n = count;

      for (let i = 0; i < this._inputDim; i++) {
        inputMean.push(this._inputWelford.mean[i]);
        const variance = n > 1 ? this._inputWelford.m2[i] / (n - 1) : 0;
        inputStd.push(Math.sqrt(variance));
      }

      for (let i = 0; i < this._outputDim; i++) {
        outputMean.push(this._outputWelford.mean[i]);
        const variance = n > 1 ? this._outputWelford.m2[i] / (n - 1) : 0;
        outputStd.push(Math.sqrt(variance));
      }
    }

    return { inputMean, inputStd, outputMean, outputStd, count };
  }

  /**
   * Reset model to initial state
   * Clears all learned parameters and statistics
   *
   * @example
   * ```typescript
   * model.reset();
   * // Model is now in initial state, ready for new training
   * ```
   */
  reset(): void {
    this._isInitialized = false;
    this._inputDim = 0;
    this._outputDim = 0;
    this._spatialDim = 0;
    this._flattenedDim = 0;
    this._convLayers = [];
    this._denseLayer = null;
    this._inputWelford = null;
    this._outputWelford = null;
    this._sampleCount = 0;
    this._updateCount = 0;
    this._totalLoss = 0;
    this._converged = false;
    this._driftCount = 0;
    this._adwinBuckets = [];
    this._adwinTotal = 0;
    this._adwinCount = 0;
    this._adwinWidth = 0;
    this._activations = [];
    this._preActivations = [];
    this._gradients = [];
    this._inputBuffer = null;
    this._outputBuffer = null;
    this._targetBuffer = null;
    this._lastInput = null;
    this._cachedBeta1Power = 1;
    this._cachedBeta2Power = 1;
    this._bufferPool.clear();
  }

  /**
   * Serialize model state to JSON string
   * Includes all weights, optimizer states, and normalization statistics
   *
   * @returns JSON string containing complete model state
   * @example
   * ```typescript
   * const modelJson = model.save();
   * localStorage.setItem('cnn-model', modelJson);
   * ```
   */
  save(): string {
    const state: SerializedState = {
      config: {
        hiddenLayers: this._hiddenLayers,
        convolutionsPerLayer: this._convolutionsPerLayer,
        kernelSize: this._kernelSize,
        learningRate: this._learningRate,
        warmupSteps: this._warmupSteps,
        totalSteps: this._totalSteps,
        beta1: this._beta1,
        beta2: this._beta2,
        epsilon: this._epsilon,
        regularizationStrength: this._regularizationStrength,
        convergenceThreshold: this._convergenceThreshold,
        outlierThreshold: this._outlierThreshold,
        adwinDelta: this._adwinDelta,
      },
      model: {
        isInitialized: this._isInitialized,
        inputDim: this._inputDim,
        outputDim: this._outputDim,
        sampleCount: this._sampleCount,
        updateCount: this._updateCount,
        totalLoss: this._totalLoss,
        converged: this._converged,
        driftCount: this._driftCount,
      },
      weights: null,
      normalization: null,
      adwin: null,
    };

    if (this._isInitialized) {
      // Serialize weights
      state.weights = {
        convLayers: this._convLayers.map((layer) => ({
          kernels: Array.from(layer.kernels),
          biases: Array.from(layer.biases),
          adamM: Array.from(layer.adamM),
          adamV: Array.from(layer.adamV),
          adamBiasM: Array.from(layer.adamBiasM),
          adamBiasV: Array.from(layer.adamBiasV),
          inputChannels: layer.inputChannels,
          outputChannels: layer.outputChannels,
        })),
        denseLayer: this._denseLayer
          ? {
            weights: Array.from(this._denseLayer.weights),
            biases: Array.from(this._denseLayer.biases),
            adamM: Array.from(this._denseLayer.adamM),
            adamV: Array.from(this._denseLayer.adamV),
            adamBiasM: Array.from(this._denseLayer.adamBiasM),
            adamBiasV: Array.from(this._denseLayer.adamBiasV),
            inputDim: this._denseLayer.inputDim,
            outputDim: this._denseLayer.outputDim,
          }
          : null,
      };

      // Serialize normalization
      state.normalization = {
        inputMean: Array.from(this._inputWelford!.mean),
        inputM2: Array.from(this._inputWelford!.m2),
        outputMean: Array.from(this._outputWelford!.mean),
        outputM2: Array.from(this._outputWelford!.m2),
        count: this._inputWelford!.count,
      };

      // Serialize ADWIN
      state.adwin = {
        buckets: this._adwinBuckets.map((level) =>
          level.map((b) => ({ ...b }))
        ),
        total: this._adwinTotal,
        count: this._adwinCount,
        width: this._adwinWidth,
      };
    }

    return JSON.stringify(state);
  }

  /**
   * Load model state from JSON string
   *
   * @param jsonString - JSON string from save()
   * @example
   * ```typescript
   * const modelJson = localStorage.getItem('cnn-model');
   * if (modelJson) {
   *   model.load(modelJson);
   * }
   * ```
   */
  load(jsonString: string): void {
    const state: SerializedState = JSON.parse(jsonString);

    this.reset();

    // Restore model metadata
    this._isInitialized = state.model.isInitialized;
    this._inputDim = state.model.inputDim;
    this._outputDim = state.model.outputDim;
    this._sampleCount = state.model.sampleCount;
    this._updateCount = state.model.updateCount;
    this._totalLoss = state.model.totalLoss;
    this._converged = state.model.converged;
    this._driftCount = state.model.driftCount;

    // Recompute cached beta powers
    this._cachedBeta1Power = Math.pow(this._beta1, this._updateCount);
    this._cachedBeta2Power = Math.pow(this._beta2, this._updateCount);

    if (this._isInitialized && state.weights) {
      this._spatialDim = this._inputDim;

      // Restore conv layers
      this._convLayers = state.weights.convLayers.map((layer) => ({
        kernels: new Float64Array(layer.kernels),
        biases: new Float64Array(layer.biases),
        kernelGrad: new Float64Array(layer.kernels.length),
        biasGrad: new Float64Array(layer.biases.length),
        adamM: new Float64Array(layer.adamM),
        adamV: new Float64Array(layer.adamV),
        adamBiasM: new Float64Array(layer.adamBiasM),
        adamBiasV: new Float64Array(layer.adamBiasV),
        inputChannels: layer.inputChannels,
        outputChannels: layer.outputChannels,
      }));

      // Restore dense layer
      if (state.weights.denseLayer) {
        const dl = state.weights.denseLayer;
        this._flattenedDim = dl.inputDim;
        this._denseLayer = {
          weights: new Float64Array(dl.weights),
          biases: new Float64Array(dl.biases),
          weightGrad: new Float64Array(dl.weights.length),
          biasGrad: new Float64Array(dl.biases.length),
          adamM: new Float64Array(dl.adamM),
          adamV: new Float64Array(dl.adamV),
          adamBiasM: new Float64Array(dl.adamBiasM),
          adamBiasV: new Float64Array(dl.adamBiasV),
          inputDim: dl.inputDim,
          outputDim: dl.outputDim,
        };
      }

      // Restore normalization
      if (state.normalization) {
        this._inputWelford = {
          mean: new Float64Array(state.normalization.inputMean),
          m2: new Float64Array(state.normalization.inputM2),
          count: state.normalization.count,
        };
        this._outputWelford = {
          mean: new Float64Array(state.normalization.outputMean),
          m2: new Float64Array(state.normalization.outputM2),
          count: state.normalization.count,
        };
      }

      // Restore ADWIN
      if (state.adwin) {
        this._adwinBuckets = state.adwin.buckets.map((level) =>
          level.map((b) => ({ ...b }))
        );
        this._adwinTotal = state.adwin.total;
        this._adwinCount = state.adwin.count;
        this._adwinWidth = state.adwin.width;
      }

      // Reallocate buffers
      this._activations = [];
      this._preActivations = [];
      this._gradients = [];

      this._activations.push(new Float64Array(this._spatialDim));
      this._preActivations.push(new Float64Array(this._spatialDim));

      for (let l = 0; l < this._hiddenLayers; l++) {
        const size = this._convolutionsPerLayer * this._spatialDim;
        this._activations.push(new Float64Array(size));
        this._preActivations.push(new Float64Array(size));
        this._gradients.push(new Float64Array(size));
      }

      this._activations.push(new Float64Array(this._outputDim));
      this._gradients.push(new Float64Array(this._flattenedDim));
      this._gradients.push(new Float64Array(this._outputDim));

      this._inputBuffer = new Float64Array(this._inputDim);
      this._outputBuffer = new Float64Array(this._outputDim);
      this._targetBuffer = new Float64Array(this._outputDim);
      this._lastInput = new Float64Array(this._inputDim);
    }
  }
}

export default ConvolutionalRegression;
