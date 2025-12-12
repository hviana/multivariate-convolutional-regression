/**
 * ConvolutionalRegression: Convolutional Neural Network for Multivariate Regression
 * with Incremental Online Learning, Adam Optimizer, and Z-Score Normalization
 *
 * @example
 * ```typescript
 * const model = new ConvolutionalRegression();
 * const result = model.fitOnline({
 *     xCoordinates: [[1, 2, 3], [4, 5, 6]],
 *     yCoordinates: [[10, 20], [30, 40]]
 * });
 * const predictions = model.predict(5);
 * ```
 */

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Result returned from fitOnline() after processing training samples
 */
export interface FitResult {
  /** Mean squared error loss for this batch */
  loss: number;
  /** L2 norm of the gradient */
  gradientNorm: number;
  /** Current learning rate after warmup/decay */
  effectiveLearningRate: number;
  /** Whether any sample was detected as outlier */
  isOutlier: boolean;
  /** Whether model has converged based on gradient norm */
  converged: boolean;
  /** Total samples processed so far */
  sampleIndex: number;
  /** Whether concept drift was detected by ADWIN */
  driftDetected: boolean;
}

/**
 * Single prediction with uncertainty bounds
 */
export interface SinglePrediction {
  /** Predicted output values */
  predicted: number[];
  /** Lower confidence bound (~95% CI) */
  lowerBound: number[];
  /** Upper confidence bound (~95% CI) */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Result from predict() containing predictions and model status
 */
export interface PredictionResult {
  /** Array of predictions for each future step */
  predictions: SinglePrediction[];
  /** Model accuracy: 1/(1 + average_loss) */
  accuracy: number;
  /** Total training samples seen */
  sampleCount: number;
  /** Whether model has seen enough samples for reliable predictions */
  isModelReady: boolean;
}

/**
 * Weight information including optimizer state
 */
export interface WeightInfo {
  /** Convolutional and dense layer kernels */
  kernels: number[][][];
  /** Layer biases */
  biases: number[][][];
  /** Adam first moment estimates */
  firstMoment: number[][][];
  /** Adam second moment estimates */
  secondMoment: number[][][];
  /** Number of Adam updates performed */
  updateCount: number;
}

/**
 * Z-score normalization statistics computed via Welford's algorithm
 */
export interface NormalizationStats {
  /** Running mean for input features */
  inputMean: number[];
  /** Running standard deviation for input features */
  inputStd: number[];
  /** Running mean for output targets */
  outputMean: number[];
  /** Running standard deviation for output targets */
  outputStd: number[];
  /** Number of samples used in statistics */
  count: number;
}

/**
 * Summary of model architecture and training state
 */
export interface ModelSummary {
  /** Whether the model has been initialized with data */
  isInitialized: boolean;
  /** Number of input features */
  inputDimension: number;
  /** Number of output targets */
  outputDimension: number;
  /** Number of convolutional layers */
  hiddenLayers: number;
  /** Output channels per conv layer */
  convolutionsPerLayer: number;
  /** Convolution kernel size */
  kernelSize: number;
  /** Total trainable parameters */
  totalParameters: number;
  /** Training samples processed */
  sampleCount: number;
  /** Current model accuracy */
  accuracy: number;
  /** Whether training has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

/**
 * Configuration options for ConvolutionalRegression
 */
export interface ConvolutionalRegressionConfig {
  /** Number of convolutional hidden layers (default: 2) */
  hiddenLayers?: number;
  /** Output channels per conv layer (default: 32) */
  convolutionsPerLayer?: number;
  /** Convolution kernel size (default: 3) */
  kernelSize?: number;
  /** Base learning rate for Adam (default: 0.001) */
  learningRate?: number;
  /** Linear warmup steps (default: 100) */
  warmupSteps?: number;
  /** Total steps for cosine decay (default: 10000) */
  totalSteps?: number;
  /** Adam β₁ for first moment (default: 0.9) */
  beta1?: number;
  /** Adam β₂ for second moment (default: 0.999) */
  beta2?: number;
  /** Numerical stability epsilon (default: 1e-8) */
  epsilon?: number;
  /** L2 regularization strength (default: 1e-4) */
  regularizationStrength?: number;
  /** Gradient norm threshold for convergence (default: 1e-6) */
  convergenceThreshold?: number;
  /** Z-score threshold for outlier detection (default: 3.0) */
  outlierThreshold?: number;
  /** ADWIN delta for drift detection (default: 0.002) */
  adwinDelta?: number;
}

// ============================================================================
// Internal Types
// ============================================================================

interface LayerConfig {
  inChannels: number;
  outChannels: number;
  width: number;
}

interface SerializedWelfordStats {
  mean: number[];
  m2: number[];
  count: number;
}

interface SerializedState {
  config: Required<ConvolutionalRegressionConfig>;
  initialized: boolean;
  inputDim: number;
  outputDim: number;
  sampleCount: number;
  updateCount: number;
  converged: boolean;
  driftCount: number;
  runningLossSum: number;
  runningLossCount: number;
  convKernels: number[][];
  convBiases: number[][];
  convKernelM: number[][];
  convKernelV: number[][];
  convBiasM: number[][];
  convBiasV: number[][];
  denseWeights: number[] | null;
  denseBias: number[] | null;
  denseWeightsM: number[] | null;
  denseWeightsV: number[] | null;
  denseBiasM: number[] | null;
  denseBiasV: number[] | null;
  inputStats: SerializedWelfordStats | null;
  outputStats: SerializedWelfordStats | null;
  residualSum: number[] | null;
  residualSqSum: number[] | null;
  layerConfigs: LayerConfig[];
  flattenSize: number;
}

// ============================================================================
// Welford's Online Statistics
// ============================================================================

/**
 * Welford's online algorithm for computing mean and variance
 *
 * Formula:
 * - δ = x - μ
 * - μ += δ/n
 * - M₂ += δ(x - μ)
 * - σ² = M₂/(n-1)
 *
 * Provides numerically stable one-pass computation of statistics
 */
class WelfordStats {
  private _mean: Float64Array;
  private _m2: Float64Array;
  private _count: number = 0;
  private readonly _dim: number;

  constructor(dim: number) {
    this._dim = dim;
    this._mean = new Float64Array(dim);
    this._m2 = new Float64Array(dim);
  }

  /**
   * Update statistics with a new sample
   * @param values - New observation vector
   */
  update(values: Float64Array): void {
    this._count++;
    const n = this._count;
    const invN = 1 / n;
    const mean = this._mean;
    const m2 = this._m2;
    const len = this._dim;

    for (let i = 0; i < len; i++) {
      const x = values[i];
      const delta = x - mean[i];
      mean[i] += delta * invN;
      const delta2 = x - mean[i];
      m2[i] += delta * delta2;
    }
  }

  /**
   * Normalize values in-place: x̃ = (x - μ)/(σ + ε)
   * @param values - Input values
   * @param output - Output buffer for normalized values
   * @param epsilon - Numerical stability constant
   */
  normalize(values: Float64Array, output: Float64Array, epsilon: number): void {
    const count = this._count;
    const mean = this._mean;
    const m2 = this._m2;
    const len = this._dim;

    if (count < 2) {
      for (let i = 0; i < len; i++) {
        output[i] = values[i];
      }
      return;
    }

    const countM1 = count - 1;
    for (let i = 0; i < len; i++) {
      const std = Math.sqrt(m2[i] / countM1 + epsilon);
      output[i] = (values[i] - mean[i]) / std;
    }
  }

  /**
   * Denormalize values: x = x̃ · σ + μ
   * @param values - Normalized values
   * @param output - Output buffer for denormalized values
   * @param epsilon - Numerical stability constant
   */
  denormalize(
    values: Float64Array,
    output: Float64Array,
    epsilon: number,
  ): void {
    const count = this._count;
    const mean = this._mean;
    const m2 = this._m2;
    const len = this._dim;

    if (count < 2) {
      for (let i = 0; i < len; i++) {
        output[i] = values[i];
      }
      return;
    }

    const countM1 = count - 1;
    for (let i = 0; i < len; i++) {
      const std = Math.sqrt(m2[i] / countM1 + epsilon);
      output[i] = values[i] * std + mean[i];
    }
  }

  /**
   * Get standard deviation array
   * @param epsilon - Numerical stability constant
   * @returns Float64Array of standard deviations
   */
  getStd(epsilon: number): Float64Array {
    const std = new Float64Array(this._dim);
    const count = this._count;
    const m2 = this._m2;

    if (count > 1) {
      const countM1 = count - 1;
      for (let i = 0; i < this._dim; i++) {
        std[i] = Math.sqrt(m2[i] / countM1 + epsilon);
      }
    } else {
      std.fill(1);
    }
    return std;
  }

  getMean(): Float64Array {
    return this._mean;
  }
  getM2(): Float64Array {
    return this._m2;
  }
  getCount(): number {
    return this._count;
  }

  /**
   * Restore state from serialized data
   */
  restore(mean: Float64Array, m2: Float64Array, count: number): void {
    this._mean.set(mean);
    this._m2.set(m2);
    this._count = count;
  }

  reset(): void {
    this._mean.fill(0);
    this._m2.fill(0);
    this._count = 0;
  }
}

// ============================================================================
// ADWIN Drift Detection
// ============================================================================

/**
 * ADWIN (ADaptive WINdowing) algorithm for concept drift detection
 *
 * Maintains a variable-length window of recent observations and detects
 * when the distribution has changed significantly.
 *
 * Drift condition: |μ₀ - μ₁| ≥ εcut where εcut = √((1/2m)ln(4n/δ))
 */
class ADWIN {
  private readonly _delta: number;
  private _bucket: Float64Array;
  private _bucketSize: number = 0;
  private _sum: number = 0;
  private _width: number = 0;
  private readonly _minWindowSize: number = 5;
  private readonly _maxWindowSize: number = 1000;

  constructor(delta: number = 0.002) {
    this._delta = delta;
    this._bucket = new Float64Array(this._maxWindowSize);
  }

  /**
   * Add a new value and check for drift
   * @param value - New observation (typically loss value)
   * @returns true if drift was detected
   */
  update(value: number): boolean {
    // Add to bucket
    if (this._width >= this._maxWindowSize) {
      // Shift window: remove oldest
      this._sum -= this._bucket[0];
      for (let i = 0; i < this._width - 1; i++) {
        this._bucket[i] = this._bucket[i + 1];
      }
      this._width--;
    }

    this._bucket[this._width] = value;
    this._sum += value;
    this._width++;

    return this._checkDrift();
  }

  private _checkDrift(): boolean {
    const width = this._width;
    const minWin = this._minWindowSize;

    if (width < minWin * 2) return false;

    const bucket = this._bucket;
    const totalSum = this._sum;
    const delta = this._delta;
    const logTerm = Math.log(4 * width / delta);

    let sum0 = 0;
    const maxSplit = width - minWin;

    for (let splitIdx = minWin; splitIdx <= maxSplit; splitIdx++) {
      sum0 += bucket[splitIdx - 1];
      const n0 = splitIdx;
      const n1 = width - splitIdx;
      const mean0 = sum0 / n0;
      const mean1 = (totalSum - sum0) / n1;

      // εcut = √((1/2m)ln(4n/δ)) where m = harmonic mean of n0, n1
      const m = (n0 * n1) / (n0 + n1);
      const epsilonCut = Math.sqrt(logTerm / (2 * m));

      if (Math.abs(mean0 - mean1) >= epsilonCut) {
        this._shrinkWindow(splitIdx);
        return true;
      }
    }
    return false;
  }

  private _shrinkWindow(splitIdx: number): void {
    const newWidth = this._width - splitIdx;
    this._sum = 0;

    for (let i = 0; i < newWidth; i++) {
      this._bucket[i] = this._bucket[splitIdx + i];
      this._sum += this._bucket[i];
    }
    this._width = newWidth;
  }

  reset(): void {
    this._sum = 0;
    this._width = 0;
  }

  getWidth(): number {
    return this._width;
  }
}

// ============================================================================
// Random Number Generation (Box-Muller)
// ============================================================================

/**
 * Thread-local Box-Muller state for generating standard normal variates
 * Uses a cached value to generate two values per computation
 */
class RandomGenerator {
  private _hasSpare: boolean = false;
  private _spare: number = 0;

  /**
   * Generate standard normal random variate using Box-Muller transform
   * @returns Random value from N(0, 1)
   */
  randn(): number {
    if (this._hasSpare) {
      this._hasSpare = false;
      return this._spare;
    }

    let u: number, v: number, s: number;
    do {
      u = Math.random() * 2 - 1;
      v = Math.random() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);

    const mul = Math.sqrt(-2 * Math.log(s) / s);
    this._spare = v * mul;
    this._hasSpare = true;
    return u * mul;
  }
}

// ============================================================================
// ConvolutionalRegression Main Class
// ============================================================================

/**
 * Convolutional Neural Network for Multivariate Regression with Online Learning
 *
 * Architecture:
 * Input(inputDim) → [Conv1D(convolutionsPerLayer, kernelSize, padding='same') → ReLU]×hiddenLayers → Flatten → Dense(outputDim)
 *
 * Features:
 * - Incremental online learning with Adam optimizer
 * - Cosine warmup learning rate schedule
 * - Welford's algorithm for z-score normalization
 * - L2 regularization
 * - Outlier detection and downweighting
 * - ADWIN concept drift detection
 * - Prediction uncertainty estimation
 *
 * @example
 * ```typescript
 * const model = new ConvolutionalRegression({
 *     hiddenLayers: 2,
 *     convolutionsPerLayer: 32,
 *     learningRate: 0.001
 * });
 *
 * // Train incrementally
 * for (const batch of dataStream) {
 *     const result = model.fitOnline(batch);
 *     console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
 * }
 *
 * // Make predictions
 * const predictions = model.predict(10);
 * ```
 */
export class ConvolutionalRegression {
  // ========================================================================
  // Configuration
  // ========================================================================

  private _hiddenLayers: number;
  private _convolutionsPerLayer: number;
  private _kernelSize: number;
  private _learningRate: number;
  private _warmupSteps: number;
  private _totalSteps: number;
  private _beta1: number;
  private _beta2: number;
  private _epsilon: number;
  private _regularizationStrength: number;
  private _convergenceThreshold: number;
  private _outlierThreshold: number;
  private _adwinDelta: number;

  // ========================================================================
  // Model State
  // ========================================================================

  private _initialized: boolean = false;
  private _inputDim: number = 0;
  private _outputDim: number = 0;
  private _sampleCount: number = 0;
  private _updateCount: number = 0;
  private _converged: boolean = false;
  private _driftCount: number = 0;
  private _runningLossSum: number = 0;
  private _runningLossCount: number = 0;

  // ========================================================================
  // Network Architecture
  // ========================================================================

  private _layerConfigs: LayerConfig[] = [];
  private _flattenSize: number = 0;

  // ========================================================================
  // Weights (Float64Array for performance)
  // ========================================================================

  /** Conv kernels: [layer][outCh * inCh * kernelSize] */
  private _convKernels: Float64Array[] = [];
  /** Conv biases: [layer][outChannels] */
  private _convBiases: Float64Array[] = [];
  /** Dense weights: [flattenSize * outputDim] */
  private _denseWeights: Float64Array | null = null;
  /** Dense bias: [outputDim] */
  private _denseBias: Float64Array | null = null;

  // ========================================================================
  // Adam Optimizer State
  // ========================================================================

  private _convKernelM: Float64Array[] = [];
  private _convKernelV: Float64Array[] = [];
  private _convBiasM: Float64Array[] = [];
  private _convBiasV: Float64Array[] = [];
  private _denseWeightsM: Float64Array | null = null;
  private _denseWeightsV: Float64Array | null = null;
  private _denseBiasM: Float64Array | null = null;
  private _denseBiasV: Float64Array | null = null;

  // ========================================================================
  // Preallocated Buffers (minimize GC pressure)
  // ========================================================================

  /** Activations after each layer: [layer][channels * width] */
  private _activations: Float64Array[] = [];
  /** Pre-ReLU activations: [layer][channels * width] */
  private _preActivations: Float64Array[] = [];
  /** Activation gradients: [layer][channels * width] */
  private _gradActivations: Float64Array[] = [];
  /** Flattened conv output */
  private _flattenedOutput: Float64Array | null = null;
  /** Dense layer output */
  private _denseOutput: Float64Array | null = null;
  /** Gradient for flattened layer */
  private _gradFlattened: Float64Array | null = null;
  /** Kernel gradients */
  private _gradKernels: Float64Array[] = [];
  /** Bias gradients */
  private _gradBiases: Float64Array[] = [];
  /** Dense weight gradients */
  private _gradDenseWeights: Float64Array | null = null;
  /** Dense bias gradients */
  private _gradDenseBias: Float64Array | null = null;

  // ========================================================================
  // Normalization & Statistics
  // ========================================================================

  private _inputStats: WelfordStats | null = null;
  private _outputStats: WelfordStats | null = null;
  private _residualSum: Float64Array | null = null;
  private _residualSqSum: Float64Array | null = null;

  // ========================================================================
  // Temporary Buffers (reused across calls)
  // ========================================================================

  private _tempInput: Float64Array | null = null;
  private _tempOutput: Float64Array | null = null;
  private _normalizedInput: Float64Array | null = null;
  private _normalizedOutput: Float64Array | null = null;
  private _denormBuffer: Float64Array | null = null;

  // ========================================================================
  // Drift Detection
  // ========================================================================

  private _adwin: ADWIN;

  // ========================================================================
  // Random Generator
  // ========================================================================

  private readonly _rng: RandomGenerator = new RandomGenerator();

  // ========================================================================
  // Constructor
  // ========================================================================

  /**
   * Create a new ConvolutionalRegression model
   *
   * @param config - Configuration options (all optional with sensible defaults)
   *
   * @example
   * ```typescript
   * // Default configuration
   * const model = new ConvolutionalRegression();
   *
   * // Custom configuration
   * const customModel = new ConvolutionalRegression({
   *     hiddenLayers: 3,
   *     convolutionsPerLayer: 64,
   *     learningRate: 0.0005
   * });
   * ```
   */
  constructor(config: ConvolutionalRegressionConfig = {}) {
    this._hiddenLayers = config.hiddenLayers ?? 2;
    this._convolutionsPerLayer = config.convolutionsPerLayer ?? 32;
    this._kernelSize = config.kernelSize ?? 3;
    this._learningRate = config.learningRate ?? 0.001;
    this._warmupSteps = config.warmupSteps ?? 100;
    this._totalSteps = config.totalSteps ?? 10000;
    this._beta1 = config.beta1 ?? 0.9;
    this._beta2 = config.beta2 ?? 0.999;
    this._epsilon = config.epsilon ?? 1e-8;
    this._regularizationStrength = config.regularizationStrength ?? 1e-4;
    this._convergenceThreshold = config.convergenceThreshold ?? 1e-6;
    this._outlierThreshold = config.outlierThreshold ?? 3.0;
    this._adwinDelta = config.adwinDelta ?? 0.002;

    this._adwin = new ADWIN(this._adwinDelta);
  }

  // ========================================================================
  // Network Initialization
  // ========================================================================

  /**
   * Initialize network architecture based on input/output dimensions
   * Uses He initialization for ReLU: W ~ N(0, √(2/fan_in))
   *
   * @param inputDim - Number of input features
   * @param outputDim - Number of output targets
   */
  private _initializeNetwork(inputDim: number, outputDim: number): void {
    this._inputDim = inputDim;
    this._outputDim = outputDim;

    const hiddenLayers = this._hiddenLayers;
    const convPerLayer = this._convolutionsPerLayer;
    const kernelSize = this._kernelSize;

    // Build layer configurations
    this._layerConfigs = [];
    let inChannels = 1;
    for (let i = 0; i < hiddenLayers; i++) {
      this._layerConfigs.push({
        inChannels,
        outChannels: convPerLayer,
        width: inputDim,
      });
      inChannels = convPerLayer;
    }

    this._flattenSize = convPerLayer * inputDim;

    // Initialize conv layers
    this._convKernels = [];
    this._convBiases = [];
    this._convKernelM = [];
    this._convKernelV = [];
    this._convBiasM = [];
    this._convBiasV = [];

    for (let l = 0; l < hiddenLayers; l++) {
      const cfg = this._layerConfigs[l];
      const kernelLen = cfg.outChannels * cfg.inChannels * kernelSize;
      const fanIn = cfg.inChannels * kernelSize;
      const stdDev = Math.sqrt(2.0 / fanIn);

      const kernel = new Float64Array(kernelLen);
      for (let i = 0; i < kernelLen; i++) {
        kernel[i] = this._rng.randn() * stdDev;
      }
      this._convKernels.push(kernel);
      this._convBiases.push(new Float64Array(cfg.outChannels));

      this._convKernelM.push(new Float64Array(kernelLen));
      this._convKernelV.push(new Float64Array(kernelLen));
      this._convBiasM.push(new Float64Array(cfg.outChannels));
      this._convBiasV.push(new Float64Array(cfg.outChannels));
    }

    // Initialize dense layer
    const denseLen = this._flattenSize * outputDim;
    const denseStdDev = Math.sqrt(2.0 / this._flattenSize);

    this._denseWeights = new Float64Array(denseLen);
    for (let i = 0; i < denseLen; i++) {
      this._denseWeights[i] = this._rng.randn() * denseStdDev;
    }
    this._denseBias = new Float64Array(outputDim);

    this._denseWeightsM = new Float64Array(denseLen);
    this._denseWeightsV = new Float64Array(denseLen);
    this._denseBiasM = new Float64Array(outputDim);
    this._denseBiasV = new Float64Array(outputDim);

    // Allocate activation buffers
    this._allocateBuffers();

    // Initialize normalization
    this._inputStats = new WelfordStats(inputDim);
    this._outputStats = new WelfordStats(outputDim);
    this._residualSum = new Float64Array(outputDim);
    this._residualSqSum = new Float64Array(outputDim);

    // Temporary buffers
    this._tempInput = new Float64Array(inputDim);
    this._tempOutput = new Float64Array(outputDim);
    this._normalizedInput = new Float64Array(inputDim);
    this._normalizedOutput = new Float64Array(outputDim);
    this._denormBuffer = new Float64Array(outputDim);

    this._initialized = true;
  }

  /**
   * Allocate all forward/backward pass buffers
   */
  private _allocateBuffers(): void {
    const hiddenLayers = this._hiddenLayers;
    const convPerLayer = this._convolutionsPerLayer;
    const inputDim = this._inputDim;
    const outputDim = this._outputDim;
    const kernelSize = this._kernelSize;

    this._activations = [];
    this._preActivations = [];
    this._gradActivations = [];

    // Input activation (1 channel)
    this._activations.push(new Float64Array(inputDim));

    for (let l = 0; l < hiddenLayers; l++) {
      const size = convPerLayer * inputDim;
      this._activations.push(new Float64Array(size));
      this._preActivations.push(new Float64Array(size));
      this._gradActivations.push(new Float64Array(size));
    }

    this._flattenedOutput = new Float64Array(this._flattenSize);
    this._denseOutput = new Float64Array(outputDim);
    this._gradFlattened = new Float64Array(this._flattenSize);

    // Gradient buffers
    this._gradKernels = [];
    this._gradBiases = [];
    for (let l = 0; l < hiddenLayers; l++) {
      const cfg = this._layerConfigs[l];
      this._gradKernels.push(
        new Float64Array(cfg.outChannels * cfg.inChannels * kernelSize),
      );
      this._gradBiases.push(new Float64Array(cfg.outChannels));
    }
    this._gradDenseWeights = new Float64Array(this._flattenSize * outputDim);
    this._gradDenseBias = new Float64Array(outputDim);
  }

  // ========================================================================
  // Forward Pass
  // ========================================================================

  /**
   * Forward propagation through the network
   *
   * Conv1D formula: y[c,i] = Σₖ Σⱼ(W[c,k,j] · x[k,i+j-pad]) + b[c]
   * ReLU: max(0, x)
   * Dense: y = Wx + b
   *
   * @param input - Normalized input features
   * @returns Output predictions (in normalized space)
   */
  private _forward(input: Float64Array): Float64Array {
    const kernelSize = this._kernelSize;
    const padding = kernelSize >> 1;
    const width = this._inputDim;

    // Copy input to first activation buffer
    this._activations[0].set(input);

    // Forward through conv layers
    const numLayers = this._layerConfigs.length;
    for (let l = 0; l < numLayers; l++) {
      const cfg = this._layerConfigs[l];
      const kernel = this._convKernels[l];
      const bias = this._convBiases[l];
      const inputAct = this._activations[l];
      const preAct = this._preActivations[l];
      const outputAct = this._activations[l + 1];

      // Conv1D with same padding
      this._conv1dForward(
        inputAct,
        kernel,
        bias,
        preAct,
        cfg.inChannels,
        cfg.outChannels,
        width,
        kernelSize,
        padding,
      );

      // ReLU activation: max(0, x)
      const len = preAct.length;
      for (let i = 0; i < len; i++) {
        outputAct[i] = preAct[i] > 0 ? preAct[i] : 0;
      }
    }

    // Flatten (already contiguous in memory)
    const lastAct = this._activations[numLayers];
    this._flattenedOutput!.set(lastAct);

    // Dense layer
    this._denseForward(
      this._flattenedOutput!,
      this._denseWeights!,
      this._denseBias!,
      this._denseOutput!,
      this._flattenSize,
      this._outputDim,
    );

    return this._denseOutput!;
  }

  /**
   * Conv1D forward pass with same padding
   *
   * y[oc, i] = Σ_ic Σ_k W[oc, ic, k] · x[ic, i + k - pad] + b[oc]
   *
   * @param input - Input tensor [inChannels * width]
   * @param kernel - Kernel tensor [outChannels * inChannels * kernelSize]
   * @param bias - Bias vector [outChannels]
   * @param output - Output tensor [outChannels * width]
   */
  private _conv1dForward(
    input: Float64Array,
    kernel: Float64Array,
    bias: Float64Array,
    output: Float64Array,
    inChannels: number,
    outChannels: number,
    width: number,
    kernelSize: number,
    padding: number,
  ): void {
    // Zero output
    output.fill(0);

    for (let oc = 0; oc < outChannels; oc++) {
      const biasVal = bias[oc];
      const ocOffset = oc * width;

      for (let i = 0; i < width; i++) {
        let sum = biasVal;

        for (let ic = 0; ic < inChannels; ic++) {
          const icOffset = ic * width;
          const kernelOffset = (oc * inChannels + ic) * kernelSize;

          for (let k = 0; k < kernelSize; k++) {
            const inputIdx = i + k - padding;
            if (inputIdx >= 0 && inputIdx < width) {
              sum += kernel[kernelOffset + k] * input[icOffset + inputIdx];
            }
          }
        }
        output[ocOffset + i] = sum;
      }
    }
  }

  /**
   * Dense layer forward: y = Wx + b
   */
  private _denseForward(
    input: Float64Array,
    weights: Float64Array,
    bias: Float64Array,
    output: Float64Array,
    inputSize: number,
    outputSize: number,
  ): void {
    for (let o = 0; o < outputSize; o++) {
      let sum = bias[o];
      const wOffset = o * inputSize;

      for (let i = 0; i < inputSize; i++) {
        sum += weights[wOffset + i] * input[i];
      }
      output[o] = sum;
    }
  }

  // ========================================================================
  // Backward Pass
  // ========================================================================

  /**
   * Backpropagation through the network
   *
   * Computes gradients via chain rule:
   * - ∂L/∂W for conv layers: correlate input with upstream gradient
   * - ∂L/∂x for conv layers: full convolution with rotated kernel
   *
   * @param target - Target values (normalized)
   * @param prediction - Network prediction (normalized)
   * @returns Gradient L2 norm
   */
  private _backward(target: Float64Array, prediction: Float64Array): number {
    const kernelSize = this._kernelSize;
    const padding = kernelSize >> 1;
    const width = this._inputDim;
    const outputDim = this._outputDim;
    const regStrength = this._regularizationStrength;

    // Compute output gradient: ∂L/∂y = (ŷ - y)
    const gradOutput = this._gradDenseBias!;
    for (let i = 0; i < outputDim; i++) {
      gradOutput[i] = prediction[i] - target[i];
    }

    // Dense layer backward
    this._denseBackward(
      this._flattenedOutput!,
      this._denseWeights!,
      gradOutput,
      this._gradDenseWeights!,
      this._gradDenseBias!,
      this._gradFlattened!,
      this._flattenSize,
      outputDim,
    );

    // Add L2 regularization gradient: ∂(λ/2 ||W||²)/∂W = λW
    const denseWeights = this._denseWeights!;
    const gradDenseWeights = this._gradDenseWeights!;
    const denseLen = denseWeights.length;
    for (let i = 0; i < denseLen; i++) {
      gradDenseWeights[i] += regStrength * denseWeights[i];
    }

    // Copy gradient to last conv layer
    const numLayers = this._layerConfigs.length;
    this._gradActivations[numLayers - 1].set(this._gradFlattened!);

    // Backward through conv layers
    for (let l = numLayers - 1; l >= 0; l--) {
      const cfg = this._layerConfigs[l];
      const kernel = this._convKernels[l];
      const inputAct = this._activations[l];
      const preAct = this._preActivations[l];
      const gradOut = this._gradActivations[l];

      // ReLU backward: ∂ReLU/∂x = 1 if x > 0 else 0
      const len = preAct.length;
      for (let i = 0; i < len; i++) {
        if (preAct[i] <= 0) {
          gradOut[i] = 0;
        }
      }

      // Get input gradient buffer (null for first layer)
      const gradIn = l > 0 ? this._gradActivations[l - 1] : null;

      // Conv1D backward
      this._conv1dBackward(
        inputAct,
        kernel,
        gradOut,
        this._gradKernels[l],
        this._gradBiases[l],
        gradIn,
        cfg.inChannels,
        cfg.outChannels,
        width,
        kernelSize,
        padding,
      );

      // Add L2 regularization gradient
      const convKernel = this._convKernels[l];
      const gradKernel = this._gradKernels[l];
      const kernelLen = convKernel.length;
      for (let i = 0; i < kernelLen; i++) {
        gradKernel[i] += regStrength * convKernel[i];
      }
    }

    // Compute gradient norm
    return this._computeGradientNorm();
  }

  /**
   * Dense layer backward pass
   *
   * ∂L/∂b = ∂L/∂y
   * ∂L/∂W = ∂L/∂y ⊗ x
   * ∂L/∂x = Wᵀ · ∂L/∂y
   */
  private _denseBackward(
    input: Float64Array,
    weights: Float64Array,
    gradOutput: Float64Array,
    gradWeights: Float64Array,
    gradBias: Float64Array,
    gradInput: Float64Array,
    inputSize: number,
    outputSize: number,
  ): void {
    // ∂L/∂b = ∂L/∂y
    gradBias.set(gradOutput);

    // ∂L/∂W[o, i] = ∂L/∂y[o] · x[i]
    for (let o = 0; o < outputSize; o++) {
      const go = gradOutput[o];
      const wOffset = o * inputSize;
      for (let i = 0; i < inputSize; i++) {
        gradWeights[wOffset + i] = go * input[i];
      }
    }

    // ∂L/∂x[i] = Σ_o W[o, i] · ∂L/∂y[o]
    gradInput.fill(0);
    for (let i = 0; i < inputSize; i++) {
      let sum = 0;
      for (let o = 0; o < outputSize; o++) {
        sum += weights[o * inputSize + i] * gradOutput[o];
      }
      gradInput[i] = sum;
    }
  }

  /**
   * Conv1D backward pass
   *
   * ∂L/∂W[oc, ic, k] = Σ_i ∂L/∂y[oc, i] · x[ic, i + k - pad]
   * ∂L/∂x[ic, i] = Σ_oc Σ_k ∂L/∂y[oc, i - k + pad] · W[oc, ic, k]
   * ∂L/∂b[oc] = Σ_i ∂L/∂y[oc, i]
   */
  private _conv1dBackward(
    input: Float64Array,
    kernel: Float64Array,
    gradOutput: Float64Array,
    gradKernel: Float64Array,
    gradBias: Float64Array,
    gradInput: Float64Array | null,
    inChannels: number,
    outChannels: number,
    width: number,
    kernelSize: number,
    padding: number,
  ): void {
    gradKernel.fill(0);
    gradBias.fill(0);
    if (gradInput) gradInput.fill(0);

    for (let oc = 0; oc < outChannels; oc++) {
      const ocOffset = oc * width;

      // Bias gradient: ∂L/∂b[oc] = Σ_i ∂L/∂y[oc, i]
      let biasGrad = 0;
      for (let i = 0; i < width; i++) {
        biasGrad += gradOutput[ocOffset + i];
      }
      gradBias[oc] = biasGrad;

      for (let ic = 0; ic < inChannels; ic++) {
        const icOffset = ic * width;
        const kernelOffset = (oc * inChannels + ic) * kernelSize;

        // Kernel gradient
        for (let k = 0; k < kernelSize; k++) {
          let kGrad = 0;
          for (let i = 0; i < width; i++) {
            const inputIdx = i + k - padding;
            if (inputIdx >= 0 && inputIdx < width) {
              kGrad += gradOutput[ocOffset + i] * input[icOffset + inputIdx];
            }
          }
          gradKernel[kernelOffset + k] = kGrad;
        }

        // Input gradient (full convolution)
        if (gradInput) {
          for (let i = 0; i < width; i++) {
            let iGrad = 0;
            for (let k = 0; k < kernelSize; k++) {
              const outputIdx = i - k + padding;
              if (outputIdx >= 0 && outputIdx < width) {
                iGrad += gradOutput[ocOffset + outputIdx] *
                  kernel[kernelOffset + k];
              }
            }
            gradInput[icOffset + i] += iGrad;
          }
        }
      }
    }
  }

  /**
   * Compute L2 norm of all gradients
   */
  private _computeGradientNorm(): number {
    let normSq = 0;

    // Conv layer gradients
    for (let l = 0; l < this._gradKernels.length; l++) {
      const gk = this._gradKernels[l];
      const len = gk.length;
      for (let i = 0; i < len; i++) {
        normSq += gk[i] * gk[i];
      }

      const gb = this._gradBiases[l];
      const bLen = gb.length;
      for (let i = 0; i < bLen; i++) {
        normSq += gb[i] * gb[i];
      }
    }

    // Dense gradients
    const gdw = this._gradDenseWeights!;
    const gdwLen = gdw.length;
    for (let i = 0; i < gdwLen; i++) {
      normSq += gdw[i] * gdw[i];
    }

    const gdb = this._gradDenseBias!;
    const gdbLen = gdb.length;
    for (let i = 0; i < gdbLen; i++) {
      normSq += gdb[i] * gdb[i];
    }

    return Math.sqrt(normSq);
  }

  // ========================================================================
  // Adam Optimizer
  // ========================================================================

  /**
   * Update weights using Adam optimizer with cosine warmup
   *
   * Learning rate schedule:
   * - Warmup (t ≤ warmupSteps): lr = baseLR × (t / warmupSteps)
   * - Decay (t > warmupSteps): lr = baseLR × 0.5 × (1 + cos(π × progress))
   *
   * Adam update:
   * - m = β₁m + (1-β₁)g
   * - v = β₂v + (1-β₂)g²
   * - m̂ = m/(1-β₁ᵗ)
   * - v̂ = v/(1-β₂ᵗ)
   * - W -= η × m̂/(√v̂ + ε)
   *
   * @returns Effective learning rate used
   */
  private _updateWeights(): number {
    this._updateCount++;
    const t = this._updateCount;

    // Compute learning rate with warmup and cosine decay
    let lr: number;
    if (t <= this._warmupSteps) {
      lr = this._learningRate * (t / this._warmupSteps);
    } else {
      const progress = (t - this._warmupSteps) /
        (this._totalSteps - this._warmupSteps);
      const decay = 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1)));
      lr = this._learningRate * decay;
    }

    const beta1 = this._beta1;
    const beta2 = this._beta2;
    const epsilon = this._epsilon;

    // Bias correction factors: 1 - βᵗ
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    // Update conv layers
    const numLayers = this._convKernels.length;
    for (let l = 0; l < numLayers; l++) {
      this._adamUpdate(
        this._convKernels[l],
        this._gradKernels[l],
        this._convKernelM[l],
        this._convKernelV[l],
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        epsilon,
      );
      this._adamUpdate(
        this._convBiases[l],
        this._gradBiases[l],
        this._convBiasM[l],
        this._convBiasV[l],
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        epsilon,
      );
    }

    // Update dense layer
    this._adamUpdate(
      this._denseWeights!,
      this._gradDenseWeights!,
      this._denseWeightsM!,
      this._denseWeightsV!,
      lr,
      beta1,
      beta2,
      bc1,
      bc2,
      epsilon,
    );
    this._adamUpdate(
      this._denseBias!,
      this._gradDenseBias!,
      this._denseBiasM!,
      this._denseBiasV!,
      lr,
      beta1,
      beta2,
      bc1,
      bc2,
      epsilon,
    );

    return lr;
  }

  /**
   * Adam update for a single parameter array (in-place)
   */
  private _adamUpdate(
    params: Float64Array,
    grads: Float64Array,
    m: Float64Array,
    v: Float64Array,
    lr: number,
    beta1: number,
    beta2: number,
    bc1: number,
    bc2: number,
    epsilon: number,
  ): void {
    const len = params.length;
    const oneMinusBeta1 = 1 - beta1;
    const oneMinusBeta2 = 1 - beta2;

    for (let i = 0; i < len; i++) {
      const g = grads[i];

      // Update biased first moment: m = β₁m + (1-β₁)g
      m[i] = beta1 * m[i] + oneMinusBeta1 * g;

      // Update biased second moment: v = β₂v + (1-β₂)g²
      v[i] = beta2 * v[i] + oneMinusBeta2 * g * g;

      // Bias-corrected estimates
      const mHat = m[i] / bc1;
      const vHat = v[i] / bc2;

      // Update: W -= η × m̂/(√v̂ + ε)
      params[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
    }
  }

  /**
   * Scale all gradients by a factor (for outlier downweighting)
   */
  private _scaleGradients(factor: number): void {
    for (let l = 0; l < this._gradKernels.length; l++) {
      const gk = this._gradKernels[l];
      const len = gk.length;
      for (let i = 0; i < len; i++) {
        gk[i] *= factor;
      }

      const gb = this._gradBiases[l];
      const bLen = gb.length;
      for (let i = 0; i < bLen; i++) {
        gb[i] *= factor;
      }
    }

    const gdw = this._gradDenseWeights!;
    const gdwLen = gdw.length;
    for (let i = 0; i < gdwLen; i++) {
      gdw[i] *= factor;
    }

    const gdb = this._gradDenseBias!;
    const gdbLen = gdb.length;
    for (let i = 0; i < gdbLen; i++) {
      gdb[i] *= factor;
    }
  }

  // ========================================================================
  // Loss and Metrics
  // ========================================================================

  /**
   * Compute prediction standard deviation from residual statistics
   */
  private _computePredictionStd(): Float64Array {
    const std = new Float64Array(this._outputDim);
    const count = this._sampleCount;
    const epsilon = this._epsilon;

    if (count > 1) {
      const residualSum = this._residualSum!;
      const residualSqSum = this._residualSqSum!;

      for (let i = 0; i < this._outputDim; i++) {
        const mean = residualSum[i] / count;
        const variance = (residualSqSum[i] / count) - (mean * mean);
        std[i] = Math.sqrt(Math.max(0, variance) + epsilon);
      }
    } else {
      std.fill(1);
    }

    return std;
  }

  /**
   * Update residual statistics for uncertainty estimation
   */
  private _updateResidualStats(
    prediction: Float64Array,
    target: Float64Array,
  ): void {
    const residualSum = this._residualSum!;
    const residualSqSum = this._residualSqSum!;
    const outputDim = this._outputDim;

    for (let i = 0; i < outputDim; i++) {
      const residual = prediction[i] - target[i];
      residualSum[i] += residual;
      residualSqSum[i] += residual * residual;
    }
  }

  /**
   * Compute accuracy metric: accuracy = 1/(1 + L̄)
   */
  private _computeAccuracy(): number {
    if (this._runningLossCount === 0) return 0;
    const avgLoss = this._runningLossSum / this._runningLossCount;
    return 1 / (1 + avgLoss);
  }

  /**
   * Compute current learning rate based on schedule
   */
  private _computeCurrentLR(): number {
    const t = this._updateCount;
    if (t === 0) return this._learningRate;

    if (t <= this._warmupSteps) {
      return this._learningRate * (t / this._warmupSteps);
    }

    const progress = (t - this._warmupSteps) /
      (this._totalSteps - this._warmupSteps);
    const decay = 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1)));
    return this._learningRate * decay;
  }

  // ========================================================================
  // Public API
  // ========================================================================

  /**
   * Fit model incrementally using online learning
   *
   * Processes samples one at a time, updating:
   * - Normalization statistics (Welford's algorithm)
   * - Network weights (Adam optimizer)
   * - Drift detection (ADWIN)
   * - Convergence status
   *
   * @param data - Training data with input coordinates and target outputs
   * @returns FitResult containing loss, gradient info, and convergence status
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *     xCoordinates: [[1, 2, 3], [4, 5, 6]],
   *     yCoordinates: [[10, 20], [30, 40]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  public fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    if (xCoordinates.length === 0 || yCoordinates.length === 0) {
      throw new Error("Empty training data");
    }

    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error("Mismatched input and output array lengths");
    }

    // Auto-detect dimensions and initialize if needed
    if (!this._initialized) {
      this._initializeNetwork(xCoordinates[0].length, yCoordinates[0].length);
    }

    const inputDim = this._inputDim;
    const outputDim = this._outputDim;
    const epsilon = this._epsilon;
    const outlierThreshold = this._outlierThreshold;
    const regStrength = this._regularizationStrength;

    let totalLoss = 0;
    let totalGradNorm = 0;
    let isOutlier = false;
    let driftDetected = false;
    let effectiveLR = 0;

    const numSamples = xCoordinates.length;

    for (let s = 0; s < numSamples; s++) {
      const x = xCoordinates[s];
      const y = yCoordinates[s];

      this._sampleCount++;

      // Copy to typed arrays
      const tempInput = this._tempInput!;
      const tempOutput = this._tempOutput!;
      for (let i = 0; i < inputDim; i++) {
        tempInput[i] = x[i];
      }
      for (let i = 0; i < outputDim; i++) {
        tempOutput[i] = y[i];
      }

      // Update normalization statistics (Welford's algorithm)
      this._inputStats!.update(tempInput);
      this._outputStats!.update(tempOutput);

      // Normalize inputs: x̃ = (x - μ)/(σ + ε)
      const normalizedInput = this._normalizedInput!;
      const normalizedOutput = this._normalizedOutput!;
      this._inputStats!.normalize(tempInput, normalizedInput, epsilon);
      this._outputStats!.normalize(tempOutput, normalizedOutput, epsilon);

      // Forward pass
      const prediction = this._forward(normalizedInput);

      // Compute MSE loss: L = (1/2)Σ‖y - ŷ‖²
      let loss = 0;
      for (let i = 0; i < outputDim; i++) {
        const diff = prediction[i] - normalizedOutput[i];
        loss += diff * diff;
      }
      loss *= 0.5;

      // Add L2 regularization loss: (λ/2)Σ‖W‖²
      let regLoss = 0;
      for (let l = 0; l < this._convKernels.length; l++) {
        const kernel = this._convKernels[l];
        const kLen = kernel.length;
        for (let i = 0; i < kLen; i++) {
          regLoss += kernel[i] * kernel[i];
        }
      }
      const denseWeights = this._denseWeights!;
      const dwLen = denseWeights.length;
      for (let i = 0; i < dwLen; i++) {
        regLoss += denseWeights[i] * denseWeights[i];
      }
      loss += 0.5 * regStrength * regLoss;

      // Check for outliers: r = (y - ŷ)/σ; |r| > threshold → outlier
      const predStd = this._computePredictionStd();
      let isCurrentOutlier = false;
      for (let i = 0; i < outputDim; i++) {
        const residual = Math.abs(prediction[i] - normalizedOutput[i]);
        const std = predStd[i];
        if (residual / std > outlierThreshold) {
          isCurrentOutlier = true;
          break;
        }
      }

      // Downweight outliers by 0.1×
      let weight = 1.0;
      if (isCurrentOutlier) {
        weight = 0.1;
        isOutlier = true;
      }

      // Backward pass
      const gradNorm = this._backward(normalizedOutput, prediction);

      // Scale gradients for outlier downweighting
      if (weight !== 1.0) {
        this._scaleGradients(weight);
      }

      // Update weights using Adam
      effectiveLR = this._updateWeights();

      // Update running loss for accuracy
      this._runningLossSum += loss;
      this._runningLossCount++;

      // Update residual statistics
      this._updateResidualStats(prediction, normalizedOutput);

      // Check for drift using ADWIN
      if (this._adwin.update(loss)) {
        driftDetected = true;
        this._driftCount++;
      }

      totalLoss += loss;
      totalGradNorm += gradNorm;
    }

    // Average results
    const avgLoss = totalLoss / numSamples;
    const avgGradNorm = totalGradNorm / numSamples;

    // Check convergence based on gradient norm
    this._converged = avgGradNorm < this._convergenceThreshold;

    return {
      loss: avgLoss,
      gradientNorm: avgGradNorm,
      effectiveLearningRate: effectiveLR,
      isOutlier,
      converged: this._converged,
      sampleIndex: this._sampleCount,
      driftDetected,
    };
  }

  /**
   * Generate predictions for future steps
   *
   * Returns predictions with uncertainty bounds computed from
   * residual statistics (approximately 95% confidence interval).
   *
   * @param futureSteps - Number of future predictions to generate
   * @returns PredictionResult with predictions and uncertainty estimates
   *
   * @example
   * ```typescript
   * const result = model.predict(10);
   * for (const pred of result.predictions) {
   *     console.log(`Predicted: ${pred.predicted}, CI: [${pred.lowerBound}, ${pred.upperBound}]`);
   * }
   * ```
   */
  public predict(futureSteps: number): PredictionResult {
    if (!this._initialized) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: 0,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const accuracy = this._computeAccuracy();
    const epsilon = this._epsilon;
    const outputDim = this._outputDim;

    // Use last normalized input or zeros
    const inputBuffer = this._normalizedInput
      ? new Float64Array(this._normalizedInput)
      : new Float64Array(this._inputDim);

    // Get prediction uncertainty
    const predStd = this._computePredictionStd();
    const outputStd = this._outputStats!.getStd(epsilon);

    // Reusable buffer for denormalization
    const denormBuffer = this._denormBuffer!;

    for (let step = 0; step < futureSteps; step++) {
      // Forward pass
      const normalizedPred = this._forward(inputBuffer);

      // Denormalize predictions
      this._outputStats!.denormalize(normalizedPred, denormBuffer, epsilon);

      const predicted: number[] = new Array(outputDim);
      const lowerBound: number[] = new Array(outputDim);
      const upperBound: number[] = new Array(outputDim);
      const standardError: number[] = new Array(outputDim);

      for (let i = 0; i < outputDim; i++) {
        predicted[i] = denormBuffer[i];

        // Standard error in output space
        const se = predStd[i] * outputStd[i];
        standardError[i] = se;

        // 95% confidence interval (±2σ)
        lowerBound[i] = denormBuffer[i] - 2 * se;
        upperBound[i] = denormBuffer[i] + 2 * se;
      }

      predictions.push({
        predicted,
        lowerBound,
        upperBound,
        standardError,
      });
    }

    return {
      predictions,
      accuracy,
      sampleCount: this._sampleCount,
      isModelReady: this._sampleCount >= this._warmupSteps,
    };
  }

  /**
   * Get summary of model architecture and training state
   *
   * @returns ModelSummary with comprehensive model information
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Parameters: ${summary.totalParameters}`);
   * console.log(`Accuracy: ${summary.accuracy}`);
   * ```
   */
  public getModelSummary(): ModelSummary {
    let totalParameters = 0;

    if (this._initialized) {
      // Conv layers
      for (let l = 0; l < this._convKernels.length; l++) {
        totalParameters += this._convKernels[l].length;
        totalParameters += this._convBiases[l].length;
      }
      // Dense layer
      totalParameters += this._denseWeights!.length;
      totalParameters += this._denseBias!.length;
    }

    return {
      isInitialized: this._initialized,
      inputDimension: this._inputDim,
      outputDimension: this._outputDim,
      hiddenLayers: this._hiddenLayers,
      convolutionsPerLayer: this._convolutionsPerLayer,
      kernelSize: this._kernelSize,
      totalParameters,
      sampleCount: this._sampleCount,
      accuracy: this._computeAccuracy(),
      converged: this._converged,
      effectiveLearningRate: this._computeCurrentLR(),
      driftCount: this._driftCount,
    };
  }

  /**
   * Get all weights and optimizer state
   *
   * @returns WeightInfo with kernels, biases, and Adam moments
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * console.log(`Update count: ${weights.updateCount}`);
   * ```
   */
  public getWeights(): WeightInfo {
    const kernels: number[][][] = [];
    const biases: number[][][] = [];
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];

    // Conv layers
    for (let l = 0; l < this._convKernels.length; l++) {
      kernels.push([Array.from(this._convKernels[l])]);
      biases.push([Array.from(this._convBiases[l])]);
      firstMoment.push([Array.from(this._convKernelM[l])]);
      secondMoment.push([Array.from(this._convKernelV[l])]);
    }

    // Dense layer
    if (this._denseWeights) {
      kernels.push([Array.from(this._denseWeights)]);
      biases.push([Array.from(this._denseBias!)]);
      firstMoment.push([Array.from(this._denseWeightsM!)]);
      secondMoment.push([Array.from(this._denseWeightsV!)]);
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
   * Get normalization statistics
   *
   * @returns NormalizationStats with mean/std for inputs and outputs
   *
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   * console.log(`Input mean: ${stats.inputMean}`);
   * ```
   */
  public getNormalizationStats(): NormalizationStats {
    if (!this._initialized || !this._inputStats || !this._outputStats) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }

    return {
      inputMean: Array.from(this._inputStats.getMean()),
      inputStd: Array.from(this._inputStats.getStd(this._epsilon)),
      outputMean: Array.from(this._outputStats.getMean()),
      outputStd: Array.from(this._outputStats.getStd(this._epsilon)),
      count: this._inputStats.getCount(),
    };
  }

  /**
   * Reset model to uninitialized state
   *
   * Clears all weights, statistics, and training history.
   *
   * @example
   * ```typescript
   * model.reset();
   * // Model is now ready to be trained on new data
   * ```
   */
  public reset(): void {
    this._initialized = false;
    this._inputDim = 0;
    this._outputDim = 0;
    this._sampleCount = 0;
    this._updateCount = 0;
    this._converged = false;
    this._driftCount = 0;
    this._runningLossSum = 0;
    this._runningLossCount = 0;

    this._layerConfigs = [];
    this._flattenSize = 0;

    this._convKernels = [];
    this._convBiases = [];
    this._convKernelM = [];
    this._convKernelV = [];
    this._convBiasM = [];
    this._convBiasV = [];
    this._denseWeights = null;
    this._denseBias = null;
    this._denseWeightsM = null;
    this._denseWeightsV = null;
    this._denseBiasM = null;
    this._denseBiasV = null;

    this._activations = [];
    this._preActivations = [];
    this._gradActivations = [];
    this._flattenedOutput = null;
    this._denseOutput = null;
    this._gradFlattened = null;
    this._gradKernels = [];
    this._gradBiases = [];
    this._gradDenseWeights = null;
    this._gradDenseBias = null;

    this._inputStats = null;
    this._outputStats = null;
    this._residualSum = null;
    this._residualSqSum = null;

    this._tempInput = null;
    this._tempOutput = null;
    this._normalizedInput = null;
    this._normalizedOutput = null;
    this._denormBuffer = null;

    this._adwin.reset();
  }

  /**
   * Serialize model state to JSON string
   *
   * Includes all weights, optimizer state, normalization statistics,
   * and training history for complete model restoration.
   *
   * @returns JSON string containing all model state
   *
   * @example
   * ```typescript
   * const savedModel = model.save();
   * localStorage.setItem('model', savedModel);
   * ```
   */
  public save(): string {
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
      initialized: this._initialized,
      inputDim: this._inputDim,
      outputDim: this._outputDim,
      sampleCount: this._sampleCount,
      updateCount: this._updateCount,
      converged: this._converged,
      driftCount: this._driftCount,
      runningLossSum: this._runningLossSum,
      runningLossCount: this._runningLossCount,

      convKernels: this._convKernels.map((k) => Array.from(k)),
      convBiases: this._convBiases.map((b) => Array.from(b)),
      convKernelM: this._convKernelM.map((m) => Array.from(m)),
      convKernelV: this._convKernelV.map((v) => Array.from(v)),
      convBiasM: this._convBiasM.map((m) => Array.from(m)),
      convBiasV: this._convBiasV.map((v) => Array.from(v)),

      denseWeights: this._denseWeights ? Array.from(this._denseWeights) : null,
      denseBias: this._denseBias ? Array.from(this._denseBias) : null,
      denseWeightsM: this._denseWeightsM
        ? Array.from(this._denseWeightsM)
        : null,
      denseWeightsV: this._denseWeightsV
        ? Array.from(this._denseWeightsV)
        : null,
      denseBiasM: this._denseBiasM ? Array.from(this._denseBiasM) : null,
      denseBiasV: this._denseBiasV ? Array.from(this._denseBiasV) : null,

      inputStats: this._inputStats
        ? {
          mean: Array.from(this._inputStats.getMean()),
          m2: Array.from(this._inputStats.getM2()),
          count: this._inputStats.getCount(),
        }
        : null,
      outputStats: this._outputStats
        ? {
          mean: Array.from(this._outputStats.getMean()),
          m2: Array.from(this._outputStats.getM2()),
          count: this._outputStats.getCount(),
        }
        : null,

      residualSum: this._residualSum ? Array.from(this._residualSum) : null,
      residualSqSum: this._residualSqSum
        ? Array.from(this._residualSqSum)
        : null,

      layerConfigs: this._layerConfigs,
      flattenSize: this._flattenSize,
    };

    return JSON.stringify(state);
  }

  /**
   * Restore model state from JSON string
   *
   * Completely restores all weights, optimizer state, normalization
   * statistics, and training history from a previously saved model.
   *
   * @param jsonStr - JSON string from save()
   *
   * @example
   * ```typescript
   * const savedModel = localStorage.getItem('model');
   * if (savedModel) {
   *     model.load(savedModel);
   * }
   * ```
   */
  public load(jsonStr: string): void {
    const state: SerializedState = JSON.parse(jsonStr);

    // Restore configuration
    this._hiddenLayers = state.config.hiddenLayers;
    this._convolutionsPerLayer = state.config.convolutionsPerLayer;
    this._kernelSize = state.config.kernelSize;
    this._learningRate = state.config.learningRate;
    this._warmupSteps = state.config.warmupSteps;
    this._totalSteps = state.config.totalSteps;
    this._beta1 = state.config.beta1;
    this._beta2 = state.config.beta2;
    this._epsilon = state.config.epsilon;
    this._regularizationStrength = state.config.regularizationStrength;
    this._convergenceThreshold = state.config.convergenceThreshold;
    this._outlierThreshold = state.config.outlierThreshold;
    this._adwinDelta = state.config.adwinDelta;

    // Restore state
    this._initialized = state.initialized;
    this._inputDim = state.inputDim;
    this._outputDim = state.outputDim;
    this._sampleCount = state.sampleCount;
    this._updateCount = state.updateCount;
    this._converged = state.converged;
    this._driftCount = state.driftCount;
    this._runningLossSum = state.runningLossSum;
    this._runningLossCount = state.runningLossCount;

    // Restore weights
    this._convKernels = state.convKernels.map((k) => new Float64Array(k));
    this._convBiases = state.convBiases.map((b) => new Float64Array(b));
    this._convKernelM = state.convKernelM.map((m) => new Float64Array(m));
    this._convKernelV = state.convKernelV.map((v) => new Float64Array(v));
    this._convBiasM = state.convBiasM.map((m) => new Float64Array(m));
    this._convBiasV = state.convBiasV.map((v) => new Float64Array(v));

    this._denseWeights = state.denseWeights
      ? new Float64Array(state.denseWeights)
      : null;
    this._denseBias = state.denseBias
      ? new Float64Array(state.denseBias)
      : null;
    this._denseWeightsM = state.denseWeightsM
      ? new Float64Array(state.denseWeightsM)
      : null;
    this._denseWeightsV = state.denseWeightsV
      ? new Float64Array(state.denseWeightsV)
      : null;
    this._denseBiasM = state.denseBiasM
      ? new Float64Array(state.denseBiasM)
      : null;
    this._denseBiasV = state.denseBiasV
      ? new Float64Array(state.denseBiasV)
      : null;

    // Restore normalization stats
    if (state.inputStats && this._inputDim > 0) {
      this._inputStats = new WelfordStats(this._inputDim);
      this._inputStats.restore(
        new Float64Array(state.inputStats.mean),
        new Float64Array(state.inputStats.m2),
        state.inputStats.count,
      );
    }
    if (state.outputStats && this._outputDim > 0) {
      this._outputStats = new WelfordStats(this._outputDim);
      this._outputStats.restore(
        new Float64Array(state.outputStats.mean),
        new Float64Array(state.outputStats.m2),
        state.outputStats.count,
      );
    }

    this._residualSum = state.residualSum
      ? new Float64Array(state.residualSum)
      : null;
    this._residualSqSum = state.residualSqSum
      ? new Float64Array(state.residualSqSum)
      : null;

    this._layerConfigs = state.layerConfigs;
    this._flattenSize = state.flattenSize;

    // Reallocate buffers if initialized
    if (this._initialized) {
      this._allocateBuffers();
      this._tempInput = new Float64Array(this._inputDim);
      this._tempOutput = new Float64Array(this._outputDim);
      this._normalizedInput = new Float64Array(this._inputDim);
      this._normalizedOutput = new Float64Array(this._outputDim);
      this._denormBuffer = new Float64Array(this._outputDim);
    }

    // Reset ADWIN
    this._adwin = new ADWIN(this._adwinDelta);
  }
}

export default ConvolutionalRegression;
