/**
 * ConvolutionalRegression - High-performance TypeScript library for convolutional
 * neural network based multivariate regression with incremental online learning.
 *
 * Features:
 * - Conv1D layers with same padding and ReLU activation
 * - Adam optimizer with cosine warmup learning rate schedule
 * - Welford's algorithm for online z-score normalization
 * - L2 regularization, outlier downweighting, ADWIN drift detection
 *
 * @module ConvolutionalRegression
 * @version 1.0.0
 */

// ============================================================================
// Type Definitions & Interfaces
// ============================================================================

/**
 * Configuration options for ConvolutionalRegression model
 */
export interface ConvolutionalRegressionConfig {
  /** Number of hidden convolutional layers (1-10). Default: 2 */
  hiddenLayers?: number;
  /** Number of convolution filters per layer (1-256). Default: 32 */
  convolutionsPerLayer?: number;
  /** Convolution kernel size. Default: 3 */
  kernelSize?: number;
  /** Base learning rate for Adam optimizer. Default: 0.001 */
  learningRate?: number;
  /** Warmup steps for learning rate schedule. Default: 100 */
  warmupSteps?: number;
  /** Total steps for cosine decay schedule. Default: 10000 */
  totalSteps?: number;
  /** Adam β₁ parameter (first moment decay). Default: 0.9 */
  beta1?: number;
  /** Adam β₂ parameter (second moment decay). Default: 0.999 */
  beta2?: number;
  /** Numerical stability constant. Default: 1e-8 */
  epsilon?: number;
  /** L2 regularization strength. Default: 1e-4 */
  regularizationStrength?: number;
  /** Loss threshold for convergence detection. Default: 1e-6 */
  convergenceThreshold?: number;
  /** Z-score threshold for outlier detection. Default: 3.0 */
  outlierThreshold?: number;
  /** Delta parameter for ADWIN drift detection. Default: 0.002 */
  adwinDelta?: number;
}

/**
 * Result returned from online training step
 */
export interface FitResult {
  /** Current MSE loss value */
  loss: number;
  /** L2 norm of gradient vector */
  gradientNorm: number;
  /** Effective learning rate after warmup/decay */
  effectiveLearningRate: number;
  /** Whether sample was flagged as outlier */
  isOutlier: boolean;
  /** Whether model has converged */
  converged: boolean;
  /** Index of most recently processed sample */
  sampleIndex: number;
  /** Whether concept drift was detected */
  driftDetected: boolean;
}

/**
 * Single prediction with uncertainty quantification
 */
export interface SinglePrediction {
  /** Point estimate of predicted values */
  predicted: number[];
  /** Lower bound of 95% confidence interval */
  lowerBound: number[];
  /** Upper bound of 95% confidence interval */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Result of prediction operation
 */
export interface PredictionResult {
  /** Predictions for each requested future step */
  predictions: SinglePrediction[];
  /** Model accuracy: 1/(1 + avgLoss) */
  accuracy: number;
  /** Total training samples processed */
  sampleCount: number;
  /** Whether model is ready for prediction */
  isModelReady: boolean;
}

/**
 * Complete weight information for serialization/inspection
 */
export interface WeightInfo {
  /** Conv layer kernels [layer][outCh][inCh*kSize] */
  kernels: number[][][];
  /** Layer biases [layer][channel] */
  biases: number[][];
  /** Adam first moment estimates */
  firstMoment: number[][][];
  /** Adam second moment estimates */
  secondMoment: number[][][];
  /** Number of Adam updates performed */
  updateCount: number;
}

/**
 * Normalization statistics from Welford's algorithm
 */
export interface NormalizationStats {
  /** Running mean of input features */
  inputMean: number[];
  /** Running std of input features */
  inputStd: number[];
  /** Running mean of output features */
  outputMean: number[];
  /** Running std of output features */
  outputStd: number[];
  /** Sample count for statistics */
  count: number;
}

/**
 * Model configuration and state summary
 */
export interface ModelSummary {
  /** Whether network has been initialized */
  isInitialized: boolean;
  /** Auto-detected input dimension */
  inputDimension: number;
  /** Auto-detected output dimension */
  outputDimension: number;
  /** Number of hidden conv layers */
  hiddenLayers: number;
  /** Filters per conv layer */
  convolutionsPerLayer: number;
  /** Convolution kernel size */
  kernelSize: number;
  /** Total trainable parameters */
  totalParameters: number;
  /** Samples processed */
  sampleCount: number;
  /** Current accuracy metric */
  accuracy: number;
  /** Whether training converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Detected drift events */
  driftCount: number;
}

/**
 * Input data structure for training
 */
export interface FitInput {
  /** Input features: [numSamples][inputDim] */
  xCoordinates: number[][];
  /** Target outputs: [numSamples][outputDim] */
  yCoordinates: number[][];
}

/**
 * Public interface for ConvolutionalRegression
 */
export interface IConvolutionalRegression {
  fitOnline(data: FitInput): FitResult;
  predict(futureSteps: number): PredictionResult;
  getModelSummary(): ModelSummary;
  getWeights(): WeightInfo;
  getNormalizationStats(): NormalizationStats;
  reset(): void;
}

// ============================================================================
// Main Implementation
// ============================================================================

/**
 * ConvolutionalRegression: High-performance CNN for multivariate regression
 * with incremental online learning capabilities.
 *
 * Architecture:
 * Input(inputDim) → [Conv1D(filters, kernelSize, same) → ReLU]×L → Flatten → Dense(outputDim)
 *
 * @example
 * ```typescript
 * const model = new ConvolutionalRegression({
 *   hiddenLayers: 2,
 *   convolutionsPerLayer: 32,
 *   learningRate: 0.001
 * });
 *
 * // Incremental training
 * const result = model.fitOnline({
 *   xCoordinates: [[1, 2, 3], [4, 5, 6]],
 *   yCoordinates: [[7, 8], [9, 10]]
 * });
 *
 * // Generate predictions
 * const predictions = model.predict(5);
 * console.log(predictions.predictions[0].predicted);
 * ```
 */
export class ConvolutionalRegression implements IConvolutionalRegression {
  // ========================================================================
  // Configuration (immutable after construction)
  // ========================================================================

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

  // ========================================================================
  // Network Dimensions (lazy initialized)
  // ========================================================================

  private _inputDim: number;
  private _outputDim: number;
  private _isInitialized: boolean;
  private _flattenSize: number;

  // ========================================================================
  // Network Parameters - Preallocated Float64Arrays
  // ========================================================================

  /** Conv kernels[layer]: flattened (outCh × inCh × kernelSize) */
  private _convKernels: Float64Array[];
  /** Conv biases[layer]: (outChannels) */
  private _convBiases: Float64Array[];
  /** Dense weights: flattened (outputDim × flattenSize) */
  private _denseWeights: Float64Array;
  /** Dense biases: (outputDim) */
  private _denseBiases: Float64Array;

  // ========================================================================
  // Adam Optimizer State
  // ========================================================================

  /** First moment for conv kernels */
  private _convKernelM: Float64Array[];
  /** Second moment for conv kernels */
  private _convKernelV: Float64Array[];
  /** First moment for conv biases */
  private _convBiasM: Float64Array[];
  /** Second moment for conv biases */
  private _convBiasV: Float64Array[];
  /** First moment for dense weights */
  private _denseWeightM: Float64Array;
  /** Second moment for dense weights */
  private _denseWeightV: Float64Array;
  /** First moment for dense biases */
  private _denseBiasM: Float64Array;
  /** Second moment for dense biases */
  private _denseBiasV: Float64Array;

  // ========================================================================
  // Welford's Algorithm State for Normalization
  // ========================================================================

  /** Running mean for inputs */
  private _inputMean: Float64Array;
  /** Running M₂ for inputs (variance computation) */
  private _inputM2: Float64Array;
  /** Running mean for outputs */
  private _outputMean: Float64Array;
  /** Running M₂ for outputs */
  private _outputM2: Float64Array;
  /** Running mean for prediction errors */
  private _errorMean: Float64Array;
  /** Running M₂ for prediction errors */
  private _errorM2: Float64Array;

  // ========================================================================
  // Training State
  // ========================================================================

  private _sampleCount: number;
  private _updateCount: number;
  private _runningLoss: number;
  private _lossCount: number;
  private _converged: boolean;
  private _driftCount: number;
  private _effectiveLr: number;

  // ========================================================================
  // ADWIN Drift Detection State
  // ========================================================================

  private _adwinWindow: Float64Array;
  private _adwinSize: number;
  private readonly _adwinMaxSize: number;

  // ========================================================================
  // Preallocated Computation Buffers
  // ========================================================================

  /** Layer activations (post-ReLU): [layer][channels × spatial] */
  private _activations: Float64Array[];
  /** Pre-activation values (pre-ReLU) */
  private _preActivations: Float64Array[];
  /** Gradient buffers for backprop */
  private _gradients: Float64Array[];
  /** Temporary gradient storage for conv kernels */
  private _gradKernelTemp: Float64Array[];
  /** Temporary gradient storage for conv biases */
  private _gradBiasTemp: Float64Array[];
  /** Temporary gradient storage for dense weights */
  private _gradDenseW: Float64Array;
  /** Temporary gradient storage for dense biases */
  private _gradDenseB: Float64Array;
  /** Normalized input buffer */
  private _normalizedInput: Float64Array;
  /** Normalized target buffer */
  private _normalizedTarget: Float64Array;
  /** Prediction output buffer */
  private _prediction: Float64Array;
  /** Last input sample (for autoregressive prediction) */
  private _lastInput: Float64Array;
  /** Dense output gradient buffer */
  private _denseOutputGrad: Float64Array;
  /** Cached channel counts per layer */
  private _inChannels: Int32Array;
  /** Padding for convolution */
  private _pad: number;

  /**
   * Creates a new ConvolutionalRegression instance
   *
   * @param config - Configuration options
   *
   * @example
   * ```typescript
   * const model = new ConvolutionalRegression({
   *   hiddenLayers: 3,
   *   convolutionsPerLayer: 64,
   *   kernelSize: 5,
   *   learningRate: 0.0005
   * });
   * ```
   */
  constructor(config: ConvolutionalRegressionConfig = {}) {
    // Validate and clamp configuration
    this._hiddenLayers = Math.max(1, Math.min(10, config.hiddenLayers ?? 2));
    this._convolutionsPerLayer = Math.max(
      1,
      Math.min(256, config.convolutionsPerLayer ?? 32),
    );
    this._kernelSize = Math.max(1, config.kernelSize ?? 3);
    this._learningRate = Math.max(1e-10, config.learningRate ?? 0.001);
    this._warmupSteps = Math.max(0, config.warmupSteps ?? 100);
    this._totalSteps = Math.max(1, config.totalSteps ?? 10000);
    this._beta1 = Math.max(0, Math.min(0.9999, config.beta1 ?? 0.9));
    this._beta2 = Math.max(0, Math.min(0.9999, config.beta2 ?? 0.999));
    this._epsilon = Math.max(1e-15, config.epsilon ?? 1e-8);
    this._regularizationStrength = Math.max(
      0,
      config.regularizationStrength ?? 1e-4,
    );
    this._convergenceThreshold = Math.max(
      0,
      config.convergenceThreshold ?? 1e-6,
    );
    this._outlierThreshold = Math.max(0, config.outlierThreshold ?? 3.0);
    this._adwinDelta = Math.max(1e-10, Math.min(1, config.adwinDelta ?? 0.002));
    this._adwinMaxSize = 1000;

    // Initialize to empty state
    this._inputDim = 0;
    this._outputDim = 0;
    this._flattenSize = 0;
    this._isInitialized = false;
    this._pad = (this._kernelSize - 1) >> 1;

    this._convKernels = [];
    this._convBiases = [];
    this._denseWeights = new Float64Array(0);
    this._denseBiases = new Float64Array(0);

    this._convKernelM = [];
    this._convKernelV = [];
    this._convBiasM = [];
    this._convBiasV = [];
    this._denseWeightM = new Float64Array(0);
    this._denseWeightV = new Float64Array(0);
    this._denseBiasM = new Float64Array(0);
    this._denseBiasV = new Float64Array(0);

    this._inputMean = new Float64Array(0);
    this._inputM2 = new Float64Array(0);
    this._outputMean = new Float64Array(0);
    this._outputM2 = new Float64Array(0);
    this._errorMean = new Float64Array(0);
    this._errorM2 = new Float64Array(0);

    this._sampleCount = 0;
    this._updateCount = 0;
    this._runningLoss = 0;
    this._lossCount = 0;
    this._converged = false;
    this._driftCount = 0;
    this._effectiveLr = 0;

    this._adwinWindow = new Float64Array(this._adwinMaxSize);
    this._adwinSize = 0;

    this._activations = [];
    this._preActivations = [];
    this._gradients = [];
    this._gradKernelTemp = [];
    this._gradBiasTemp = [];
    this._gradDenseW = new Float64Array(0);
    this._gradDenseB = new Float64Array(0);
    this._normalizedInput = new Float64Array(0);
    this._normalizedTarget = new Float64Array(0);
    this._prediction = new Float64Array(0);
    this._lastInput = new Float64Array(0);
    this._denseOutputGrad = new Float64Array(0);
    this._inChannels = new Int32Array(0);
  }

  /**
   * Performs incremental online learning on provided data.
   * Processes each sample sequentially with Adam optimizer.
   *
   * Algorithm:
   * 1. Normalize inputs: x̃ = (x - μ)/(σ + ε) using Welford's running stats
   * 2. Forward: propagate through conv layers with ReLU activation
   * 3. Loss: L = (1/2n)Σ‖y - ŷ‖² + (λ/2)Σ‖W‖²
   * 4. Backprop: compute gradients via chain rule
   * 5. Adam update: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g²
   * 6. Check for outliers and concept drift
   *
   * @param data - Training data with xCoordinates and yCoordinates
   * @returns Result of the last training step
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2], [3, 4], [5, 6]],
   *   yCoordinates: [[7], [8], [9]]
   * });
   * console.log(`Loss: ${result.loss.toFixed(6)}`);
   * ```
   */
  public fitOnline(data: FitInput): FitResult {
    const { xCoordinates, yCoordinates } = data;

    // Validate input
    if (
      !xCoordinates || !yCoordinates ||
      xCoordinates.length === 0 || yCoordinates.length === 0
    ) {
      return this._emptyFitResult();
    }

    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error("xCoordinates and yCoordinates must have equal length");
    }

    // Lazy initialization on first call
    if (!this._isInitialized) {
      this._inputDim = xCoordinates[0].length;
      this._outputDim = yCoordinates[0].length;

      if (this._inputDim === 0 || this._outputDim === 0) {
        throw new Error("Input and output dimensions must be positive");
      }

      this._initializeNetwork();
    }

    // Validate dimensions
    if (
      xCoordinates[0].length !== this._inputDim ||
      yCoordinates[0].length !== this._outputDim
    ) {
      throw new Error("Input dimensions do not match initialized dimensions");
    }

    // Process samples incrementally
    let result: FitResult = this._emptyFitResult();
    const numSamples = xCoordinates.length;

    for (let idx = 0; idx < numSamples; idx++) {
      result = this._trainSample(xCoordinates[idx], yCoordinates[idx], idx);
    }

    return result;
  }

  /**
   * Generates predictions for future steps with uncertainty estimates.
   * Uses autoregressive prediction if input/output dimensions match.
   *
   * Uncertainty: computed from prediction error statistics
   * - standardError = σ_error / √n
   * - 95% CI: predicted ± 1.96 × standardError
   *
   * @param futureSteps - Number of steps to predict
   * @returns Predictions with confidence intervals
   *
   * @example
   * ```typescript
   * const result = model.predict(10);
   * for (const pred of result.predictions) {
   *   console.log(`Value: ${pred.predicted[0].toFixed(4)}`);
   *   console.log(`95% CI: [${pred.lowerBound[0].toFixed(4)}, ${pred.upperBound[0].toFixed(4)}]`);
   * }
   * ```
   */
  public predict(futureSteps: number): PredictionResult {
    if (!this._isInitialized || futureSteps <= 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this._sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const isAutoregressive = this._inputDim === this._outputDim;
    const standardError = this._computeStandardError();

    // Create working buffer for current input
    const currentInput = new Float64Array(this._inputDim);
    const currentNormalized = new Float64Array(this._inputDim);

    // Initialize with last seen input
    for (let i = 0; i < this._inputDim; i++) {
      currentInput[i] = this._lastInput[i];
    }

    for (let step = 0; step < futureSteps; step++) {
      // Normalize current input
      this._normalizeInPlace(
        currentInput,
        this._inputMean,
        this._inputM2,
        currentNormalized,
      );

      // Forward pass
      this._forward(currentNormalized);

      // Denormalize prediction
      const predicted = new Array<number>(this._outputDim);
      const lowerBound = new Array<number>(this._outputDim);
      const upperBound = new Array<number>(this._outputDim);
      const se = new Array<number>(this._outputDim);

      for (let j = 0; j < this._outputDim; j++) {
        const variance = this._sampleCount > 1
          ? this._outputM2[j] / (this._sampleCount - 1)
          : 1.0;
        const std = Math.sqrt(variance + this._epsilon);
        const pred = this._prediction[j] * std + this._outputMean[j];

        predicted[j] = pred;
        se[j] = standardError[j];
        lowerBound[j] = pred - 1.96 * se[j];
        upperBound[j] = pred + 1.96 * se[j];
      }

      predictions.push({
        predicted,
        lowerBound,
        upperBound,
        standardError: se,
      });

      // Autoregressive: use prediction as next input
      if (isAutoregressive) {
        for (let j = 0; j < this._outputDim; j++) {
          currentInput[j] = predicted[j];
        }
      }
    }

    // Accuracy: 1/(1 + avgLoss)
    const avgLoss = this._lossCount > 0
      ? this._runningLoss / this._lossCount
      : 1;
    const accuracy = 1 / (1 + avgLoss);

    return {
      predictions,
      accuracy,
      sampleCount: this._sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Returns model configuration and current state summary
   *
   * @returns Model summary
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Parameters: ${summary.totalParameters}`);
   * console.log(`Accuracy: ${(summary.accuracy * 100).toFixed(2)}%`);
   * ```
   */
  public getModelSummary(): ModelSummary {
    let totalParams = 0;

    if (this._isInitialized) {
      // Conv layer parameters
      for (let layer = 0; layer < this._hiddenLayers; layer++) {
        totalParams += this._convKernels[layer].length;
        totalParams += this._convBiases[layer].length;
      }
      // Dense layer parameters
      totalParams += this._denseWeights.length;
      totalParams += this._denseBiases.length;
    }

    const avgLoss = this._lossCount > 0
      ? this._runningLoss / this._lossCount
      : 1;

    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inputDim,
      outputDimension: this._outputDim,
      hiddenLayers: this._hiddenLayers,
      convolutionsPerLayer: this._convolutionsPerLayer,
      kernelSize: this._kernelSize,
      totalParameters: totalParams,
      sampleCount: this._sampleCount,
      accuracy: 1 / (1 + avgLoss),
      converged: this._converged,
      effectiveLearningRate: this._effectiveLr,
      driftCount: this._driftCount,
    };
  }

  /**
   * Returns all model weights and Adam optimizer state
   *
   * @returns Weight information for serialization
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * // Serialize for storage
   * const serialized = JSON.stringify(weights);
   * ```
   */
  public getWeights(): WeightInfo {
    const kernels: number[][][] = [];
    const biases: number[][] = [];
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];

    if (!this._isInitialized) {
      return { kernels, biases, firstMoment, secondMoment, updateCount: 0 };
    }

    // Conv layers
    for (let layer = 0; layer < this._hiddenLayers; layer++) {
      const inCh = this._inChannels[layer];
      const outCh = this._convolutionsPerLayer;
      const layerKernels: number[][] = [];
      const layerM: number[][] = [];
      const layerV: number[][] = [];

      for (let o = 0; o < outCh; o++) {
        const filterSize = inCh * this._kernelSize;
        const kernelRow: number[] = new Array(filterSize);
        const mRow: number[] = new Array(filterSize);
        const vRow: number[] = new Array(filterSize);

        for (let k = 0; k < filterSize; k++) {
          const idx = o * filterSize + k;
          kernelRow[k] = this._convKernels[layer][idx];
          mRow[k] = this._convKernelM[layer][idx];
          vRow[k] = this._convKernelV[layer][idx];
        }

        layerKernels.push(kernelRow);
        layerM.push(mRow);
        layerV.push(vRow);
      }

      kernels.push(layerKernels);
      biases.push(Array.from(this._convBiases[layer]));
      firstMoment.push(layerM);
      secondMoment.push(layerV);
    }

    // Dense layer
    const denseKernels: number[][] = [];
    const denseM: number[][] = [];
    const denseV: number[][] = [];

    for (let o = 0; o < this._outputDim; o++) {
      const row: number[] = new Array(this._flattenSize);
      const mRow: number[] = new Array(this._flattenSize);
      const vRow: number[] = new Array(this._flattenSize);

      for (let i = 0; i < this._flattenSize; i++) {
        const idx = o * this._flattenSize + i;
        row[i] = this._denseWeights[idx];
        mRow[i] = this._denseWeightM[idx];
        vRow[i] = this._denseWeightV[idx];
      }

      denseKernels.push(row);
      denseM.push(mRow);
      denseV.push(vRow);
    }

    kernels.push(denseKernels);
    biases.push(Array.from(this._denseBiases));
    firstMoment.push(denseM);
    secondMoment.push(denseV);

    return {
      kernels,
      biases,
      firstMoment,
      secondMoment,
      updateCount: this._updateCount,
    };
  }

  /**
   * Returns normalization statistics from Welford's algorithm
   *
   * Welford's online algorithm:
   * - δ = x - μ
   * - μ += δ/n
   * - M₂ += δ(x - μ)
   * - σ² = M₂/(n-1)
   *
   * @returns Normalization statistics
   */
  public getNormalizationStats(): NormalizationStats {
    if (!this._isInitialized) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }

    const inputStd = new Array<number>(this._inputDim);
    const outputStd = new Array<number>(this._outputDim);
    const n = this._sampleCount;

    for (let i = 0; i < this._inputDim; i++) {
      const variance = n > 1 ? this._inputM2[i] / (n - 1) : 1;
      inputStd[i] = Math.sqrt(variance + this._epsilon);
    }

    for (let i = 0; i < this._outputDim; i++) {
      const variance = n > 1 ? this._outputM2[i] / (n - 1) : 1;
      outputStd[i] = Math.sqrt(variance + this._epsilon);
    }

    return {
      inputMean: Array.from(this._inputMean),
      inputStd,
      outputMean: Array.from(this._outputMean),
      outputStd,
      count: n,
    };
  }

  /**
   * Resets model to initial state, clearing all learned weights and statistics
   */
  public reset(): void {
    this._inputDim = 0;
    this._outputDim = 0;
    this._flattenSize = 0;
    this._isInitialized = false;

    this._convKernels = [];
    this._convBiases = [];
    this._denseWeights = new Float64Array(0);
    this._denseBiases = new Float64Array(0);

    this._convKernelM = [];
    this._convKernelV = [];
    this._convBiasM = [];
    this._convBiasV = [];
    this._denseWeightM = new Float64Array(0);
    this._denseWeightV = new Float64Array(0);
    this._denseBiasM = new Float64Array(0);
    this._denseBiasV = new Float64Array(0);

    this._inputMean = new Float64Array(0);
    this._inputM2 = new Float64Array(0);
    this._outputMean = new Float64Array(0);
    this._outputM2 = new Float64Array(0);
    this._errorMean = new Float64Array(0);
    this._errorM2 = new Float64Array(0);

    this._sampleCount = 0;
    this._updateCount = 0;
    this._runningLoss = 0;
    this._lossCount = 0;
    this._converged = false;
    this._driftCount = 0;
    this._effectiveLr = 0;
    this._adwinSize = 0;

    this._activations = [];
    this._preActivations = [];
    this._gradients = [];
    this._gradKernelTemp = [];
    this._gradBiasTemp = [];
    this._gradDenseW = new Float64Array(0);
    this._gradDenseB = new Float64Array(0);
    this._normalizedInput = new Float64Array(0);
    this._normalizedTarget = new Float64Array(0);
    this._prediction = new Float64Array(0);
    this._lastInput = new Float64Array(0);
    this._denseOutputGrad = new Float64Array(0);
    this._inChannels = new Int32Array(0);
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  /**
   * Initializes all network weights and preallocates buffers.
   * Uses He initialization: W ~ N(0, √(2/fan_in)) for ReLU networks.
   */
  private _initializeNetwork(): void {
    const kSize = this._kernelSize;
    const numConvs = this._convolutionsPerLayer;
    const spatial = this._inputDim;

    this._flattenSize = numConvs * spatial;
    this._inChannels = new Int32Array(this._hiddenLayers);

    // Initialize conv layers
    for (let layer = 0; layer < this._hiddenLayers; layer++) {
      const inCh = layer === 0 ? 1 : numConvs;
      this._inChannels[layer] = inCh;
      const outCh = numConvs;
      const kernelLen = outCh * inCh * kSize;
      const fanIn = inCh * kSize;

      // He initialization
      this._convKernels.push(this._heInit(kernelLen, fanIn));
      this._convBiases.push(new Float64Array(outCh));

      // Adam state (zeros)
      this._convKernelM.push(new Float64Array(kernelLen));
      this._convKernelV.push(new Float64Array(kernelLen));
      this._convBiasM.push(new Float64Array(outCh));
      this._convBiasV.push(new Float64Array(outCh));

      // Gradient temps
      this._gradKernelTemp.push(new Float64Array(kernelLen));
      this._gradBiasTemp.push(new Float64Array(outCh));
    }

    // Dense layer
    const denseLen = this._flattenSize * this._outputDim;
    this._denseWeights = this._heInit(denseLen, this._flattenSize);
    this._denseBiases = new Float64Array(this._outputDim);
    this._denseWeightM = new Float64Array(denseLen);
    this._denseWeightV = new Float64Array(denseLen);
    this._denseBiasM = new Float64Array(this._outputDim);
    this._denseBiasV = new Float64Array(this._outputDim);
    this._gradDenseW = new Float64Array(denseLen);
    this._gradDenseB = new Float64Array(this._outputDim);

    // Activations: layer 0 = input (1 × spatial), layers 1-L = conv (numConvs × spatial)
    this._activations.push(new Float64Array(spatial));
    this._preActivations.push(new Float64Array(spatial));

    for (let layer = 0; layer < this._hiddenLayers; layer++) {
      const size = numConvs * spatial;
      this._activations.push(new Float64Array(size));
      this._preActivations.push(new Float64Array(size));
    }

    // Gradient buffers
    this._gradients.push(new Float64Array(spatial));
    for (let layer = 0; layer < this._hiddenLayers; layer++) {
      this._gradients.push(new Float64Array(numConvs * spatial));
    }

    // Normalization stats
    this._inputMean = new Float64Array(this._inputDim);
    this._inputM2 = new Float64Array(this._inputDim);
    this._outputMean = new Float64Array(this._outputDim);
    this._outputM2 = new Float64Array(this._outputDim);
    this._errorMean = new Float64Array(this._outputDim);
    this._errorM2 = new Float64Array(this._outputDim);

    // Computation buffers
    this._normalizedInput = new Float64Array(this._inputDim);
    this._normalizedTarget = new Float64Array(this._outputDim);
    this._prediction = new Float64Array(this._outputDim);
    this._lastInput = new Float64Array(this._inputDim);
    this._denseOutputGrad = new Float64Array(this._outputDim);

    this._isInitialized = true;
  }

  /**
   * He initialization for ReLU networks.
   * W ~ N(0, √(2/fan_in)) using Box-Muller transform.
   */
  private _heInit(size: number, fanIn: number): Float64Array {
    const arr = new Float64Array(size);
    const stddev = Math.sqrt(2.0 / fanIn);

    // Box-Muller generates pairs
    for (let i = 0; i < size; i += 2) {
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      const mag = stddev * Math.sqrt(-2.0 * Math.log(u1));
      const theta = 6.283185307179586 * u2; // 2π

      arr[i] = mag * Math.cos(theta);
      if (i + 1 < size) {
        arr[i + 1] = mag * Math.sin(theta);
      }
    }

    return arr;
  }

  /**
   * Trains on a single sample with online learning.
   */
  private _trainSample(x: number[], y: number[], idx: number): FitResult {
    this._sampleCount++;
    const n = this._sampleCount;

    // Copy to typed arrays and save last input
    for (let i = 0; i < this._inputDim; i++) {
      this._lastInput[i] = x[i];
    }

    // Update Welford stats BEFORE normalizing (incremental mean/variance)
    // δ = x - μ, μ += δ/n, M₂ += δ(x - μ)
    this._updateWelford(this._lastInput, this._inputMean, this._inputM2, n);
    this._updateWelfordFromArray(y, this._outputMean, this._outputM2, n);

    // Z-score normalize: x̃ = (x - μ)/(σ + ε)
    this._normalizeInPlace(
      this._lastInput,
      this._inputMean,
      this._inputM2,
      this._normalizedInput,
    );
    this._normalizeFromArray(
      y,
      this._outputMean,
      this._outputM2,
      this._normalizedTarget,
    );

    // Forward pass
    this._forward(this._normalizedInput);

    // Compute MSE loss: L = (1/2d)Σ(ŷ - y)²
    let mse = 0;
    for (let i = 0; i < this._outputDim; i++) {
      const diff = this._prediction[i] - this._normalizedTarget[i];
      mse += diff * diff;
    }
    mse /= this._outputDim;

    // L2 regularization: L_reg = (λ/2)Σw²
    const regLoss = this._computeRegLoss();
    const totalLoss = mse + regLoss;

    // Outlier detection: r = error/σ_error, outlier if |r| > threshold
    const isOutlier = this._checkOutlier();
    const sampleWeight = isOutlier ? 0.1 : 1.0;

    // Backward pass
    const gradNorm = this._backward(sampleWeight);

    // Adam update with warmup + cosine decay
    this._effectiveLr = this._adamUpdate();

    // Update error statistics for uncertainty estimation
    this._updateErrorStats();

    // Running loss for accuracy tracking: L̄ = ΣLoss/n
    this._runningLoss += totalLoss;
    this._lossCount++;

    // Convergence check
    const avgLoss = this._runningLoss / this._lossCount;
    this._converged = avgLoss < this._convergenceThreshold;

    // ADWIN drift detection
    const driftDetected = this._adwinCheck(totalLoss);
    if (driftDetected) {
      this._driftCount++;
    }

    return {
      loss: totalLoss,
      gradientNorm: gradNorm,
      effectiveLearningRate: this._effectiveLr,
      isOutlier,
      converged: this._converged,
      sampleIndex: idx,
      driftDetected,
    };
  }

  /**
   * Full forward pass through network.
   * Conv1D: y[c,i] = Σₖ Σⱼ(W[c,k,j] · x[k,i+j-pad]) + b[c]
   * ReLU: max(0, x)
   */
  private _forward(input: Float64Array): void {
    const spatial = this._inputDim;
    const numConvs = this._convolutionsPerLayer;
    const kSize = this._kernelSize;
    const pad = this._pad;

    // Copy input to first activation layer
    const act0 = this._activations[0];
    for (let i = 0; i < spatial; i++) {
      act0[i] = input[i];
    }

    // Forward through conv layers
    for (let layer = 0; layer < this._hiddenLayers; layer++) {
      const inCh = this._inChannels[layer];
      const inputAct = this._activations[layer];
      const outputAct = this._activations[layer + 1];
      const preAct = this._preActivations[layer + 1];
      const kernel = this._convKernels[layer];
      const bias = this._convBiases[layer];

      let outIdx = 0;
      for (let oc = 0; oc < numConvs; oc++) {
        const biasVal = bias[oc];
        const kernelBase = oc * inCh * kSize;

        for (let i = 0; i < spatial; i++) {
          let sum = biasVal;

          // Convolution with same padding
          for (let ic = 0; ic < inCh; ic++) {
            const inputBase = ic * spatial;
            const kBase = kernelBase + ic * kSize;

            for (let j = 0; j < kSize; j++) {
              const inputIdx = i + j - pad;
              // Zero padding: skip out-of-bounds
              if (inputIdx >= 0 && inputIdx < spatial) {
                sum += kernel[kBase + j] * inputAct[inputBase + inputIdx];
              }
            }
          }

          preAct[outIdx] = sum;
          // ReLU activation
          outputAct[outIdx] = sum > 0 ? sum : 0;
          outIdx++;
        }
      }
    }

    // Dense layer: y = Wx + b
    const lastAct = this._activations[this._hiddenLayers];
    const flattenSize = this._flattenSize;

    for (let o = 0; o < this._outputDim; o++) {
      let sum = this._denseBiases[o];
      const weightBase = o * flattenSize;

      for (let i = 0; i < flattenSize; i++) {
        sum += this._denseWeights[weightBase + i] * lastAct[i];
      }

      this._prediction[o] = sum;
    }
  }

  /**
   * Full backward pass computing gradients.
   * ∂L/∂W via chain rule, ∂Conv: convolve with upstream gradient.
   */
  private _backward(sampleWeight: number): number {
    const spatial = this._inputDim;
    const numConvs = this._convolutionsPerLayer;
    const flattenSize = this._flattenSize;
    const kSize = this._kernelSize;
    const pad = this._pad;

    let gradNormSq = 0;

    // Output gradient: ∂L/∂ŷ = (ŷ - y) × sampleWeight
    for (let o = 0; o < this._outputDim; o++) {
      const diff = (this._prediction[o] - this._normalizedTarget[o]) *
        sampleWeight;
      this._denseOutputGrad[o] = diff;
      gradNormSq += diff * diff;
    }

    // Dense layer backward
    // ∂L/∂W = (ŷ - y) ⊗ activation
    // ∂L/∂activation = Wᵀ · (ŷ - y)
    const lastAct = this._activations[this._hiddenLayers];
    const flatGrad = this._gradients[this._hiddenLayers];

    // Zero flatten gradient
    for (let i = 0; i < flattenSize; i++) {
      flatGrad[i] = 0;
    }

    for (let o = 0; o < this._outputDim; o++) {
      const diff = this._denseOutputGrad[o];
      this._gradDenseB[o] = diff;

      const weightBase = o * flattenSize;
      for (let i = 0; i < flattenSize; i++) {
        this._gradDenseW[weightBase + i] = diff * lastAct[i];
        flatGrad[i] += diff * this._denseWeights[weightBase + i];
      }
    }

    // Backward through conv layers (reverse order)
    for (let layer = this._hiddenLayers - 1; layer >= 0; layer--) {
      const inCh = this._inChannels[layer];
      const inputAct = this._activations[layer];
      const preAct = this._preActivations[layer + 1];
      const upstream = this._gradients[layer + 1];
      const kernel = this._convKernels[layer];
      const gradKernel = this._gradKernelTemp[layer];
      const gradBias = this._gradBiasTemp[layer];
      const gradInput = this._gradients[layer];

      const kernelLen = gradKernel.length;
      const biasLen = gradBias.length;
      const inputLen = gradInput.length;

      // Zero gradients
      for (let i = 0; i < kernelLen; i++) gradKernel[i] = 0;
      for (let i = 0; i < biasLen; i++) gradBias[i] = 0;
      for (let i = 0; i < inputLen; i++) gradInput[i] = 0;

      let upstreamIdx = 0;
      for (let oc = 0; oc < numConvs; oc++) {
        const kernelBase = oc * inCh * kSize;

        for (let i = 0; i < spatial; i++) {
          // ReLU derivative: 1 if preAct > 0, else 0
          const reluGrad = preAct[upstreamIdx] > 0 ? upstream[upstreamIdx] : 0;
          upstreamIdx++;

          if (reluGrad === 0) continue;

          gradBias[oc] += reluGrad;

          for (let ic = 0; ic < inCh; ic++) {
            const inputBase = ic * spatial;
            const kBase = kernelBase + ic * kSize;

            for (let j = 0; j < kSize; j++) {
              const inputIdx = i + j - pad;
              if (inputIdx >= 0 && inputIdx < spatial) {
                // ∂L/∂W += upstream × input
                gradKernel[kBase + j] += reluGrad *
                  inputAct[inputBase + inputIdx];
                // ∂L/∂input += upstream × W (for next layer)
                gradInput[inputBase + inputIdx] += reluGrad * kernel[kBase + j];
              }
            }
          }
        }
      }
    }

    return Math.sqrt(gradNormSq);
  }

  /**
   * Adam optimizer update with warmup + cosine decay.
   *
   * Learning rate schedule:
   * - Warmup: lr = base_lr × (t / warmup_steps)
   * - Decay: lr = base_lr × 0.5 × (1 + cos(π × progress))
   *
   * Adam: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g²
   *       W -= lr × (m/(1-β₁ᵗ)) / (√(v/(1-β₂ᵗ)) + ε)
   */
  private _adamUpdate(): number {
    this._updateCount++;
    const t = this._updateCount;

    // Learning rate with warmup and cosine decay
    let lr: number;
    if (t <= this._warmupSteps) {
      lr = this._learningRate * (t / Math.max(1, this._warmupSteps));
    } else {
      const progress = (t - this._warmupSteps) /
        Math.max(1, this._totalSteps - this._warmupSteps);
      lr = this._learningRate * 0.5 *
        (1 + Math.cos(3.141592653589793 * Math.min(progress, 1)));
    }

    // Bias correction: 1 - β^t
    const beta1Corr = 1 - Math.pow(this._beta1, t);
    const beta2Corr = 1 - Math.pow(this._beta2, t);

    // Update conv layers
    for (let layer = 0; layer < this._hiddenLayers; layer++) {
      this._adamUpdateArr(
        this._convKernels[layer],
        this._gradKernelTemp[layer],
        this._convKernelM[layer],
        this._convKernelV[layer],
        lr,
        beta1Corr,
        beta2Corr,
      );
      this._adamUpdateArr(
        this._convBiases[layer],
        this._gradBiasTemp[layer],
        this._convBiasM[layer],
        this._convBiasV[layer],
        lr,
        beta1Corr,
        beta2Corr,
      );
    }

    // Update dense layer
    this._adamUpdateArr(
      this._denseWeights,
      this._gradDenseW,
      this._denseWeightM,
      this._denseWeightV,
      lr,
      beta1Corr,
      beta2Corr,
    );
    this._adamUpdateArr(
      this._denseBiases,
      this._gradDenseB,
      this._denseBiasM,
      this._denseBiasV,
      lr,
      beta1Corr,
      beta2Corr,
    );

    return lr;
  }

  /**
   * In-place Adam update for single weight array with L2 regularization.
   */
  private _adamUpdateArr(
    weights: Float64Array,
    grads: Float64Array,
    m: Float64Array,
    v: Float64Array,
    lr: number,
    beta1Corr: number,
    beta2Corr: number,
  ): void {
    const len = weights.length;
    const beta1 = this._beta1;
    const beta2 = this._beta2;
    const eps = this._epsilon;
    const lambda = this._regularizationStrength;

    for (let i = 0; i < len; i++) {
      // Add L2 regularization gradient: ∂(λ/2 w²)/∂w = λw
      const g = grads[i] + lambda * weights[i];

      // First moment: m = β₁m + (1-β₁)g
      m[i] = beta1 * m[i] + (1 - beta1) * g;

      // Second moment: v = β₂v + (1-β₂)g²
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;

      // Bias-corrected estimates
      const mHat = m[i] / beta1Corr;
      const vHat = v[i] / beta2Corr;

      // Update: W -= lr × m̂ / (√v̂ + ε)
      weights[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  /**
   * Welford's algorithm: δ = x - μ, μ += δ/n, M₂ += δ(x - μ)
   */
  private _updateWelford(
    x: Float64Array,
    mean: Float64Array,
    m2: Float64Array,
    n: number,
  ): void {
    const len = x.length;
    for (let i = 0; i < len; i++) {
      const delta = x[i] - mean[i];
      mean[i] += delta / n;
      const delta2 = x[i] - mean[i];
      m2[i] += delta * delta2;
    }
  }

  /**
   * Welford's algorithm from regular array input
   */
  private _updateWelfordFromArray(
    x: number[],
    mean: Float64Array,
    m2: Float64Array,
    n: number,
  ): void {
    const len = x.length;
    for (let i = 0; i < len; i++) {
      const delta = x[i] - mean[i];
      mean[i] += delta / n;
      const delta2 = x[i] - mean[i];
      m2[i] += delta * delta2;
    }
  }

  /**
   * Z-score normalization: x̃ = (x - μ)/(σ + ε), σ² = M₂/(n-1)
   */
  private _normalizeInPlace(
    x: Float64Array,
    mean: Float64Array,
    m2: Float64Array,
    output: Float64Array,
  ): void {
    const n = this._sampleCount;
    const eps = this._epsilon;
    const len = x.length;

    for (let i = 0; i < len; i++) {
      const variance = n > 1 ? m2[i] / (n - 1) : 1;
      const std = Math.sqrt(variance + eps);
      output[i] = (x[i] - mean[i]) / std;
    }
  }

  /**
   * Z-score normalization from regular array
   */
  private _normalizeFromArray(
    x: number[],
    mean: Float64Array,
    m2: Float64Array,
    output: Float64Array,
  ): void {
    const n = this._sampleCount;
    const eps = this._epsilon;
    const len = x.length;

    for (let i = 0; i < len; i++) {
      const variance = n > 1 ? m2[i] / (n - 1) : 1;
      const std = Math.sqrt(variance + eps);
      output[i] = (x[i] - mean[i]) / std;
    }
  }

  /**
   * Computes L2 regularization loss: L_reg = (λ/2)Σw²
   */
  private _computeRegLoss(): number {
    let regLoss = 0;

    for (let layer = 0; layer < this._hiddenLayers; layer++) {
      const kernel = this._convKernels[layer];
      const len = kernel.length;
      for (let i = 0; i < len; i++) {
        regLoss += kernel[i] * kernel[i];
      }
    }

    const denseLen = this._denseWeights.length;
    for (let i = 0; i < denseLen; i++) {
      regLoss += this._denseWeights[i] * this._denseWeights[i];
    }

    return 0.5 * this._regularizationStrength * regLoss;
  }

  /**
   * Outlier check: r = error/σ_error, outlier if |r| > threshold
   */
  private _checkOutlier(): boolean {
    if (this._sampleCount < 10) return false;

    const n = this._sampleCount;
    const eps = this._epsilon;

    for (let i = 0; i < this._outputDim; i++) {
      const variance = n > 1 ? this._errorM2[i] / (n - 1) : 1;
      const errorStd = Math.sqrt(variance + eps);
      const error = Math.abs(this._prediction[i] - this._normalizedTarget[i]);
      const zScore = errorStd > eps ? error / errorStd : 0;

      if (zScore > this._outlierThreshold) {
        return true;
      }
    }

    return false;
  }

  /**
   * Update prediction error statistics
   */
  private _updateErrorStats(): void {
    const n = this._sampleCount;

    for (let i = 0; i < this._outputDim; i++) {
      const error = this._prediction[i] - this._normalizedTarget[i];
      const delta = error - this._errorMean[i];
      this._errorMean[i] += delta / n;
      const delta2 = error - this._errorMean[i];
      this._errorM2[i] += delta * delta2;
    }
  }

  /**
   * Compute standard error for predictions: SE = σ/√n
   */
  private _computeStandardError(): Float64Array {
    const se = new Float64Array(this._outputDim);
    const n = this._sampleCount;
    const eps = this._epsilon;

    if (n <= 1) {
      for (let i = 0; i < this._outputDim; i++) {
        se[i] = 1;
      }
      return se;
    }

    for (let i = 0; i < this._outputDim; i++) {
      const variance = this._errorM2[i] / (n - 1);
      // Denormalize: multiply by output std
      const outputVar = this._outputM2[i] / (n - 1);
      const outputStd = Math.sqrt(outputVar + eps);
      se[i] = outputStd * Math.sqrt((variance + eps) / n);
    }

    return se;
  }

  /**
   * ADWIN drift detection.
   * Detect drift when |μ₀ - μ₁| ≥ εcut where εcut = √((1/2m)ln(4|W|/δ))
   */
  private _adwinCheck(error: number): boolean {
    // Add to window
    if (this._adwinSize < this._adwinMaxSize) {
      this._adwinWindow[this._adwinSize] = error;
      this._adwinSize++;
    } else {
      // Shift window
      for (let i = 0; i < this._adwinMaxSize - 1; i++) {
        this._adwinWindow[i] = this._adwinWindow[i + 1];
      }
      this._adwinWindow[this._adwinMaxSize - 1] = error;
    }

    if (this._adwinSize < 10) return false;

    // Compute total sum
    let totalSum = 0;
    for (let i = 0; i < this._adwinSize; i++) {
      totalSum += this._adwinWindow[i];
    }

    // Try all cuts to find significant difference
    let leftSum = 0;
    const delta = this._adwinDelta;

    for (let cut = 1; cut < this._adwinSize; cut++) {
      leftSum += this._adwinWindow[cut - 1];
      const rightSum = totalSum - leftSum;

      const n0 = cut;
      const n1 = this._adwinSize - cut;
      const mu0 = leftSum / n0;
      const mu1 = rightSum / n1;

      // Harmonic mean-based m
      const m = 1 / (1 / n0 + 1 / n1);
      const epsCut = Math.sqrt(
        (0.5 / m) * Math.log(4 * this._adwinSize / delta),
      );

      if (Math.abs(mu0 - mu1) >= epsCut) {
        // Drift detected - shrink window
        const newSize = this._adwinSize - cut;
        for (let i = 0; i < newSize; i++) {
          this._adwinWindow[i] = this._adwinWindow[cut + i];
        }
        this._adwinSize = newSize;
        return true;
      }
    }

    return false;
  }

  /**
   * Creates empty FitResult for error/edge cases
   */
  private _emptyFitResult(): FitResult {
    return {
      loss: 0,
      gradientNorm: 0,
      effectiveLearningRate: 0,
      isOutlier: false,
      converged: false,
      sampleIndex: -1,
      driftDetected: false,
    };
  }
}

export { ConvolutionalRegression as default };
