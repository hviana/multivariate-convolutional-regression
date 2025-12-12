/**
 * @fileoverview ConvolutionalRegression - 1D CNN for Multivariate Regression with Online Learning
 *
 * High-performance implementation using Float64Arrays, preallocated buffers,
 * and in-place operations to minimize garbage collection pressure.
 *
 * Architecture: Input → [Conv1D(same) → ReLU]×L → Flatten → Dense → Output
 *
 * @example
 * ```typescript
 * const model = new ConvolutionalRegression({ hiddenLayers: 2, convolutionsPerLayer: 32 });
 *
 * // Online learning
 * const result = model.fitOnline({ xCoordinates: [[1, 2, 3]], yCoordinates: [[4, 5]] });
 *
 * // Batch learning
 * const batchResult = model.fitBatch({ xCoordinates: data, yCoordinates: labels, epochs: 100 });
 *
 * // Prediction
 * const predictions = model.predict(5);
 * ```
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/** Configuration options for ConvolutionalRegression */
export interface ConvolutionalRegressionConfig {
  /** Number of hidden convolutional layers (1-10, default: 2) */
  hiddenLayers?: number;
  /** Number of convolution filters per layer (1-256, default: 32) */
  convolutionsPerLayer?: number;
  /** Kernel size for convolution operations (default: 3) */
  kernelSize?: number;
  /** Base learning rate for Adam optimizer (default: 0.001) */
  learningRate?: number;
  /** Number of linear warmup steps (default: 100) */
  warmupSteps?: number;
  /** Total training steps for cosine decay schedule (default: 10000) */
  totalSteps?: number;
  /** Adam exponential decay rate for first moment (default: 0.9) */
  beta1?: number;
  /** Adam exponential decay rate for second moment (default: 0.999) */
  beta2?: number;
  /** Small constant for numerical stability (default: 1e-8) */
  epsilon?: number;
  /** L2 regularization coefficient (default: 1e-4) */
  regularizationStrength?: number;
  /** Mini-batch size for batch training (default: 32) */
  batchSize?: number;
  /** Loss improvement threshold for convergence (default: 1e-6) */
  convergenceThreshold?: number;
  /** Z-score threshold for outlier detection (default: 3.0) */
  outlierThreshold?: number;
  /** ADWIN confidence parameter for drift detection (default: 0.002) */
  adwinDelta?: number;
}

/** Result from a single online training step */
export interface FitResult {
  /** Current loss value (MSE + regularization) */
  loss: number;
  /** L2 norm of the gradient */
  gradientNorm: number;
  /** Learning rate after warmup and decay */
  effectiveLearningRate: number;
  /** Whether this sample was detected as an outlier */
  isOutlier: boolean;
  /** Whether the model has converged */
  converged: boolean;
  /** Number of samples processed so far */
  sampleIndex: number;
  /** Whether concept drift was detected */
  driftDetected: boolean;
}

/** Result from batch training */
export interface BatchFitResult {
  /** Final loss after training */
  finalLoss: number;
  /** History of loss values per epoch */
  lossHistory: number[];
  /** Whether training converged early */
  converged: boolean;
  /** Number of epochs completed */
  epochsCompleted: number;
  /** Total samples processed across all epochs */
  totalSamplesProcessed: number;
}

/** A single prediction with uncertainty bounds */
export interface SinglePrediction {
  /** Predicted output values */
  predicted: number[];
  /** Lower confidence bound (mean - 2*stderr) */
  lowerBound: number[];
  /** Upper confidence bound (mean + 2*stderr) */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/** Result from prediction */
export interface PredictionResult {
  /** Array of predictions for requested future steps */
  predictions: SinglePrediction[];
  /** Model accuracy estimate: 1/(1 + avgLoss) */
  accuracy: number;
  /** Number of training samples seen */
  sampleCount: number;
  /** Whether the model has been trained */
  isModelReady: boolean;
}

/** Weight information for inspection or serialization */
export interface WeightInfo {
  /** Convolutional kernel weights [layer][outChannel][inChannel*kernelSize] */
  kernels: number[][][];
  /** Bias values [layer][outChannel] */
  biases: number[][];
  /** Adam first moment estimates */
  firstMoment: number[][][];
  /** Adam second moment estimates */
  secondMoment: number[][][];
  /** Number of Adam updates performed */
  updateCount: number;
}

/** Normalization statistics */
export interface NormalizationStats {
  /** Running mean of input features */
  inputMean: number[];
  /** Running standard deviation of input features */
  inputStd: number[];
  /** Running mean of output values */
  outputMean: number[];
  /** Running standard deviation of output values */
  outputStd: number[];
  /** Number of samples used for statistics */
  count: number;
}

/** Model summary information */
export interface ModelSummary {
  /** Whether the model has been initialized */
  isInitialized: boolean;
  /** Input feature dimension */
  inputDimension: number;
  /** Output dimension */
  outputDimension: number;
  /** Number of convolutional layers */
  hiddenLayers: number;
  /** Filters per convolutional layer */
  convolutionsPerLayer: number;
  /** Convolution kernel size */
  kernelSize: number;
  /** Total number of trainable parameters */
  totalParameters: number;
  /** Number of training samples seen */
  sampleCount: number;
  /** Current accuracy estimate */
  accuracy: number;
  /** Whether model has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

/** Public interface for ConvolutionalRegression */
export interface IConvolutionalRegression {
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult;
  fitBatch(
    data: {
      xCoordinates: number[][];
      yCoordinates: number[][];
      epochs?: number;
    },
  ): BatchFitResult;
  predict(futureSteps: number): PredictionResult;
  getModelSummary(): ModelSummary;
  getWeights(): WeightInfo;
  getNormalizationStats(): NormalizationStats;
  reset(): void;
}

// ============================================================================
// MAIN IMPLEMENTATION
// ============================================================================

/**
 * Convolutional Neural Network for Regression with Online Learning
 *
 * A 1D convolutional neural network supporting both online (incremental)
 * and batch learning. Uses Adam optimizer with cosine warmup, Welford's
 * algorithm for z-score normalization, ADWIN for drift detection.
 *
 * @implements {IConvolutionalRegression}
 */
export class ConvolutionalRegression implements IConvolutionalRegression {
  // ========================================================================
  // CONFIGURATION (immutable after construction)
  // ========================================================================

  private readonly _hiddenLayers: number;
  private readonly _convPerLayer: number;
  private readonly _kernelSize: number;
  private readonly _baseLR: number;
  private readonly _warmupSteps: number;
  private readonly _totalSteps: number;
  private readonly _beta1: number;
  private readonly _beta2: number;
  private readonly _epsilon: number;
  private readonly _lambda: number;
  private readonly _batchSize: number;
  private readonly _convThreshold: number;
  private readonly _outlierThreshold: number;
  private readonly _adwinDelta: number;

  // ========================================================================
  // NETWORK STATE
  // ========================================================================

  private _initialized: boolean = false;
  private _inputDim: number = 0;
  private _outputDim: number = 0;
  private _spatialDim: number = 0;
  private _flattenedSize: number = 0;

  private _layerInCh: Int32Array = new Int32Array(0);
  private _layerOutCh: Int32Array = new Int32Array(0);

  // ========================================================================
  // WEIGHTS AND ADAM STATE (Float64Arrays for performance)
  // ========================================================================

  // Convolutional layers: flat arrays [outCh * inCh * kernelSize]
  private _convKernels: Float64Array[] = [];
  private _convBiases: Float64Array[] = [];
  private _convKernelM: Float64Array[] = [];
  private _convKernelV: Float64Array[] = [];
  private _convBiasM: Float64Array[] = [];
  private _convBiasV: Float64Array[] = [];

  // Dense layer: [flattenedSize * outputDim]
  private _denseW: Float64Array = new Float64Array(0);
  private _denseB: Float64Array = new Float64Array(0);
  private _denseWM: Float64Array = new Float64Array(0);
  private _denseWV: Float64Array = new Float64Array(0);
  private _denseBM: Float64Array = new Float64Array(0);
  private _denseBV: Float64Array = new Float64Array(0);

  // ========================================================================
  // ACTIVATION AND GRADIENT BUFFERS (preallocated)
  // ========================================================================

  private _convInputs: Float64Array[] = [];
  private _convPreAct: Float64Array[] = [];
  private _convOutputs: Float64Array[] = [];
  private _convGrads: Float64Array[] = [];
  private _convKernelGrad: Float64Array[] = [];
  private _convBiasGrad: Float64Array[] = [];

  private _denseInput: Float64Array = new Float64Array(0);
  private _denseOutput: Float64Array = new Float64Array(0);
  private _denseGradOut: Float64Array = new Float64Array(0);
  private _denseGradIn: Float64Array = new Float64Array(0);
  private _denseWGrad: Float64Array = new Float64Array(0);
  private _denseBGrad: Float64Array = new Float64Array(0);

  // ========================================================================
  // NORMALIZATION STATE (Welford's algorithm)
  // ========================================================================

  private _inputMean: Float64Array = new Float64Array(0);
  private _inputM2: Float64Array = new Float64Array(0);
  private _outputMean: Float64Array = new Float64Array(0);
  private _outputM2: Float64Array = new Float64Array(0);
  private _normCount: number = 0;

  // ========================================================================
  // TRAINING STATE
  // ========================================================================

  private _updateCount: number = 0;
  private _sampleCount: number = 0;
  private _totalLoss: number = 0;
  private _converged: boolean = false;
  private _lastLoss: number = Infinity;

  // ========================================================================
  // ADWIN DRIFT DETECTION
  // ========================================================================

  private _adwinWindow: Float64Array = new Float64Array(0);
  private _adwinHead: number = 0;
  private _adwinSize: number = 0;
  private _adwinSum: number = 0;
  private _adwinSumSq: number = 0;
  private _driftCount: number = 0;
  private readonly _maxAdwinSize: number = 1024;

  // ========================================================================
  // UNCERTAINTY ESTIMATION
  // ========================================================================

  private _residualMean: Float64Array = new Float64Array(0);
  private _residualM2: Float64Array = new Float64Array(0);
  private _residualCount: number = 0;

  // ========================================================================
  // TEMPORARY BUFFERS (reused to avoid allocations)
  // ========================================================================

  private _tmpNorm: Float64Array = new Float64Array(0);
  private _tmpTarget: Float64Array = new Float64Array(0);
  private _lastInput: Float64Array = new Float64Array(0);
  private _lastOutput: Float64Array = new Float64Array(0);

  // Box-Muller spare
  private _hasSpare: boolean = false;
  private _spare: number = 0;

  // ========================================================================
  // CONSTRUCTOR
  // ========================================================================

  /**
   * Creates a new ConvolutionalRegression model
   *
   * @param config - Configuration options with sensible defaults
   *
   * @example
   * ```typescript
   * const model = new ConvolutionalRegression({
   *   hiddenLayers: 3,
   *   convolutionsPerLayer: 64,
   *   learningRate: 0.0005
   * });
   * ```
   */
  constructor(config: ConvolutionalRegressionConfig = {}) {
    this._hiddenLayers = Math.max(1, Math.min(10, config.hiddenLayers ?? 2));
    this._convPerLayer = Math.max(
      1,
      Math.min(256, config.convolutionsPerLayer ?? 32),
    );
    this._kernelSize = Math.max(1, config.kernelSize ?? 3);
    this._baseLR = Math.max(1e-10, config.learningRate ?? 0.001);
    this._warmupSteps = Math.max(0, config.warmupSteps ?? 100);
    this._totalSteps = Math.max(1, config.totalSteps ?? 10000);
    this._beta1 = Math.max(0, Math.min(0.9999, config.beta1 ?? 0.9));
    this._beta2 = Math.max(0, Math.min(0.9999, config.beta2 ?? 0.999));
    this._epsilon = Math.max(1e-15, config.epsilon ?? 1e-8);
    this._lambda = Math.max(0, config.regularizationStrength ?? 1e-4);
    this._batchSize = Math.max(1, config.batchSize ?? 32);
    this._convThreshold = Math.max(0, config.convergenceThreshold ?? 1e-6);
    this._outlierThreshold = Math.max(0.1, config.outlierThreshold ?? 3.0);
    this._adwinDelta = Math.max(
      1e-10,
      Math.min(0.5, config.adwinDelta ?? 0.002),
    );
  }

  // ========================================================================
  // PUBLIC API
  // ========================================================================

  /**
   * Performs one step of online (incremental) learning
   *
   * Implements Adam optimizer with cosine warmup, Welford's z-score normalization,
   * L2 regularization, outlier downweighting, and ADWIN drift detection.
   *
   * @formula Loss: L = (1/2n)Σ‖y - ŷ‖² + (λ/2)Σ‖W‖²
   * @formula Adam: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², W -= η·m̂/(√v̂ + ε)
   * @formula Welford: μₙ = μₙ₋₁ + (xₙ - μₙ₋₁)/n
   *
   * @param data - Training data with xCoordinates and yCoordinates
   * @returns FitResult with loss, gradient info, and training status
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1.0, 2.0, 3.0]],
   *   yCoordinates: [[4.0, 5.0]]
   * });
   * ```
   */
  public fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    if (!xCoordinates?.length || !yCoordinates?.length) {
      return this._emptyFitResult();
    }

    const x = xCoordinates[0];
    const y = yCoordinates[0];

    if (!x?.length || !y?.length) {
      return this._emptyFitResult();
    }

    if (!this._initialized) {
      this._initNetwork(x.length, y.length);
    }

    if (x.length !== this._inputDim || y.length !== this._outputDim) {
      return this._emptyFitResult();
    }

    this._sampleCount++;

    // Update normalization (Welford's)
    this._updateNormStats(x, y);

    // Normalize input
    this._normalizeInput(x, this._tmpNorm);

    // Forward pass
    const pred = this._forward(this._tmpNorm);

    // Normalize target
    this._normalizeOutput(y, this._tmpTarget);

    // Compute MSE loss
    let loss = 0;
    for (let i = 0; i < this._outputDim; i++) {
      const d = pred[i] - this._tmpTarget[i];
      loss += d * d;
    }
    loss = loss / (2 * this._outputDim);

    // Add L2 regularization
    loss += this._computeRegLoss();

    // Outlier detection
    const isOutlier = this._detectOutlier(pred, this._tmpTarget);
    const weight = isOutlier ? 0.1 : 1.0;

    // Update residual stats
    this._updateResidualStats(pred, this._tmpTarget);

    // Compute learning rate
    this._updateCount++;
    const lr = this._computeLR();

    // Backward pass with Adam update
    const gradNorm = this._backward(this._tmpTarget, lr, weight);

    // Update loss tracking
    this._totalLoss += loss;
    const lossDelta = Math.abs(this._lastLoss - loss);
    this._converged = lossDelta < this._convThreshold &&
      this._sampleCount > 100;
    this._lastLoss = loss;

    // ADWIN drift detection
    const driftDetected = this._updateAdwin(loss);

    // Store last values for prediction
    this._copyTo(x, this._lastInput);
    this._copyTo(y, this._lastOutput);

    return {
      loss,
      gradientNorm: gradNorm,
      effectiveLearningRate: lr,
      isOutlier,
      converged: this._converged,
      sampleIndex: this._sampleCount,
      driftDetected,
    };
  }

  /**
   * Performs batch training with mini-batch gradient descent
   *
   * Implements Fisher-Yates shuffle, mini-batch gradient accumulation,
   * and early stopping based on convergence.
   *
   * @param data - Training data and optional epoch count
   * @returns BatchFitResult with loss history and convergence info
   *
   * @example
   * ```typescript
   * const result = model.fitBatch({
   *   xCoordinates: trainX,
   *   yCoordinates: trainY,
   *   epochs: 100
   * });
   * ```
   */
  public fitBatch(data: {
    xCoordinates: number[][];
    yCoordinates: number[][];
    epochs?: number;
  }): BatchFitResult {
    const { xCoordinates, yCoordinates, epochs = 100 } = data;

    if (!xCoordinates?.length || !yCoordinates?.length) {
      return {
        finalLoss: 0,
        lossHistory: [],
        converged: false,
        epochsCompleted: 0,
        totalSamplesProcessed: 0,
      };
    }

    const n = Math.min(xCoordinates.length, yCoordinates.length);
    if (n === 0 || !xCoordinates[0]?.length || !yCoordinates[0]?.length) {
      return {
        finalLoss: 0,
        lossHistory: [],
        converged: false,
        epochsCompleted: 0,
        totalSamplesProcessed: 0,
      };
    }

    if (!this._initialized) {
      this._initNetwork(xCoordinates[0].length, yCoordinates[0].length);
    }

    // Pre-compute normalization stats
    for (let i = 0; i < n; i++) {
      this._updateNormStats(xCoordinates[i], yCoordinates[i]);
    }

    // Index array for shuffling
    const indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    const lossHistory: number[] = [];
    let totalProcessed = 0;
    let epochsCompleted = 0;
    let converged = false;
    let prevLoss = Infinity;

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Fisher-Yates shuffle
      for (let i = n - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const t = indices[i];
        indices[i] = indices[j];
        indices[j] = t;
      }

      let epochLoss = 0;
      let batchCount = 0;

      for (let start = 0; start < n; start += this._batchSize) {
        const end = Math.min(start + this._batchSize, n);
        const batchN = end - start;

        this._zeroGradients();

        let batchLoss = 0;

        for (let b = start; b < end; b++) {
          const idx = indices[b];

          this._normalizeInput(xCoordinates[idx], this._tmpNorm);
          this._normalizeOutput(yCoordinates[idx], this._tmpTarget);

          const pred = this._forward(this._tmpNorm);

          let sampleLoss = 0;
          for (let i = 0; i < this._outputDim; i++) {
            const d = pred[i] - this._tmpTarget[i];
            sampleLoss += d * d;
          }
          batchLoss += sampleLoss / (2 * this._outputDim);

          this._accumulateGrads(this._tmpTarget);
          totalProcessed++;
        }

        // Average and add regularization
        const invBatch = 1 / batchN;
        this._scaleGradients(invBatch);
        this._addRegGradients();

        // Adam update
        this._updateCount++;
        const lr = this._computeLR();
        this._applyAdam(lr);

        epochLoss += batchLoss;
        batchCount += batchN;
        this._sampleCount += batchN;
      }

      epochLoss = epochLoss / batchCount + this._computeRegLoss();
      lossHistory.push(epochLoss);
      epochsCompleted++;

      // Early stopping
      if (Math.abs(prevLoss - epochLoss) < this._convThreshold && epoch > 10) {
        converged = true;
        break;
      }
      prevLoss = epochLoss;
    }

    this._converged = converged;
    this._totalLoss = lossHistory.length > 0
      ? lossHistory[lossHistory.length - 1] * this._sampleCount
      : 0;

    return {
      finalLoss: lossHistory.length > 0
        ? lossHistory[lossHistory.length - 1]
        : 0,
      lossHistory,
      converged,
      epochsCompleted,
      totalSamplesProcessed: totalProcessed,
    };
  }

  /**
   * Makes predictions for future steps
   *
   * Uses the last seen input pattern and returns uncertainty bounds
   * based on residual statistics.
   *
   * @param futureSteps - Number of predictions to make
   * @returns PredictionResult with predictions and uncertainty estimates
   *
   * @example
   * ```typescript
   * const result = model.predict(5);
   * for (const pred of result.predictions) {
   *   console.log(`Predicted: ${pred.predicted}`);
   * }
   * ```
   */
  public predict(futureSteps: number): PredictionResult {
    if (!this._initialized || this._sampleCount === 0 || futureSteps <= 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this._sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const accuracy = this._computeAccuracy();

    // Working buffer for autoregressive prediction
    const currentInput = new Float64Array(this._inputDim);
    currentInput.set(this._lastInput.subarray(0, this._inputDim));

    for (let step = 0; step < futureSteps; step++) {
      this._normalizeInput(currentInput, this._tmpNorm);
      const normPred = this._forward(this._tmpNorm);

      const predicted: number[] = new Array(this._outputDim);
      const lowerBound: number[] = new Array(this._outputDim);
      const upperBound: number[] = new Array(this._outputDim);
      const standardError: number[] = new Array(this._outputDim);

      for (let i = 0; i < this._outputDim; i++) {
        const outStd = this._getOutputStd(i);
        predicted[i] = normPred[i] * outStd + this._outputMean[i];

        let se = 0;
        if (this._residualCount > 1) {
          const resVar = this._residualM2[i] / (this._residualCount - 1);
          se = Math.sqrt(Math.max(0, resVar)) * outStd;
        }
        standardError[i] = se;
        lowerBound[i] = predicted[i] - 2 * se;
        upperBound[i] = predicted[i] + 2 * se;
      }

      predictions.push({ predicted, lowerBound, upperBound, standardError });

      // Autoregressive: shift input and append prediction
      if (this._inputDim >= this._outputDim) {
        for (let i = 0; i < this._inputDim - this._outputDim; i++) {
          currentInput[i] = currentInput[i + this._outputDim];
        }
        for (let i = 0; i < this._outputDim; i++) {
          currentInput[this._inputDim - this._outputDim + i] = predicted[i];
        }
      }
    }

    return {
      predictions,
      accuracy,
      sampleCount: this._sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Returns a summary of the model's current state
   * @returns ModelSummary with architecture and training statistics
   */
  public getModelSummary(): ModelSummary {
    return {
      isInitialized: this._initialized,
      inputDimension: this._inputDim,
      outputDimension: this._outputDim,
      hiddenLayers: this._hiddenLayers,
      convolutionsPerLayer: this._convPerLayer,
      kernelSize: this._kernelSize,
      totalParameters: this._countParams(),
      sampleCount: this._sampleCount,
      accuracy: this._computeAccuracy(),
      converged: this._converged,
      effectiveLearningRate: this._computeLR(),
      driftCount: this._driftCount,
    };
  }

  /**
   * Returns the current weights and Adam optimizer state
   * @returns WeightInfo with all learnable parameters
   */
  public getWeights(): WeightInfo {
    const kernels: number[][][] = [];
    const biases: number[][] = [];
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];

    for (let l = 0; l < this._hiddenLayers; l++) {
      const outCh = this._layerOutCh[l];
      const inCh = this._layerInCh[l];
      const ks = this._kernelSize;

      const lk: number[][] = [];
      const lm: number[][] = [];
      const lv: number[][] = [];

      for (let c = 0; c < outCh; c++) {
        const offset = c * inCh * ks;
        const k: number[] = [];
        const m: number[] = [];
        const v: number[] = [];
        for (let i = 0; i < inCh * ks; i++) {
          k.push(this._convKernels[l][offset + i]);
          m.push(this._convKernelM[l][offset + i]);
          v.push(this._convKernelV[l][offset + i]);
        }
        lk.push(k);
        lm.push(m);
        lv.push(v);
      }

      kernels.push(lk);
      firstMoment.push(lm);
      secondMoment.push(lv);

      const lb: number[] = [];
      for (let c = 0; c < outCh; c++) lb.push(this._convBiases[l][c]);
      biases.push(lb);
    }

    // Dense layer
    const dk: number[] = [];
    const dm: number[] = [];
    const dv: number[] = [];
    for (let i = 0; i < this._denseW.length; i++) {
      dk.push(this._denseW[i]);
      dm.push(this._denseWM[i]);
      dv.push(this._denseWV[i]);
    }
    kernels.push([dk]);
    firstMoment.push([dm]);
    secondMoment.push([dv]);

    const db: number[] = [];
    for (let i = 0; i < this._denseB.length; i++) db.push(this._denseB[i]);
    biases.push(db);

    return {
      kernels,
      biases,
      firstMoment,
      secondMoment,
      updateCount: this._updateCount,
    };
  }

  /**
   * Returns the current normalization statistics
   * @returns NormalizationStats with mean and std for inputs/outputs
   */
  public getNormalizationStats(): NormalizationStats {
    const inputMean: number[] = [];
    const inputStd: number[] = [];
    const outputMean: number[] = [];
    const outputStd: number[] = [];

    for (let i = 0; i < this._inputDim; i++) {
      inputMean.push(this._inputMean[i]);
      inputStd.push(this._getInputStd(i));
    }
    for (let i = 0; i < this._outputDim; i++) {
      outputMean.push(this._outputMean[i]);
      outputStd.push(this._getOutputStd(i));
    }

    return {
      inputMean,
      inputStd,
      outputMean,
      outputStd,
      count: this._normCount,
    };
  }

  /**
   * Resets the model to its initial untrained state
   */
  public reset(): void {
    this._initialized = false;
    this._inputDim = 0;
    this._outputDim = 0;
    this._spatialDim = 0;
    this._flattenedSize = 0;

    this._layerInCh = new Int32Array(0);
    this._layerOutCh = new Int32Array(0);

    this._convKernels = [];
    this._convBiases = [];
    this._convKernelM = [];
    this._convKernelV = [];
    this._convBiasM = [];
    this._convBiasV = [];

    this._denseW = new Float64Array(0);
    this._denseB = new Float64Array(0);
    this._denseWM = new Float64Array(0);
    this._denseWV = new Float64Array(0);
    this._denseBM = new Float64Array(0);
    this._denseBV = new Float64Array(0);

    this._convInputs = [];
    this._convPreAct = [];
    this._convOutputs = [];
    this._convGrads = [];
    this._convKernelGrad = [];
    this._convBiasGrad = [];

    this._denseInput = new Float64Array(0);
    this._denseOutput = new Float64Array(0);
    this._denseGradOut = new Float64Array(0);
    this._denseGradIn = new Float64Array(0);
    this._denseWGrad = new Float64Array(0);
    this._denseBGrad = new Float64Array(0);

    this._inputMean = new Float64Array(0);
    this._inputM2 = new Float64Array(0);
    this._outputMean = new Float64Array(0);
    this._outputM2 = new Float64Array(0);
    this._normCount = 0;

    this._updateCount = 0;
    this._sampleCount = 0;
    this._totalLoss = 0;
    this._converged = false;
    this._lastLoss = Infinity;

    this._adwinWindow = new Float64Array(0);
    this._adwinHead = 0;
    this._adwinSize = 0;
    this._adwinSum = 0;
    this._adwinSumSq = 0;
    this._driftCount = 0;

    this._residualMean = new Float64Array(0);
    this._residualM2 = new Float64Array(0);
    this._residualCount = 0;

    this._tmpNorm = new Float64Array(0);
    this._tmpTarget = new Float64Array(0);
    this._lastInput = new Float64Array(0);
    this._lastOutput = new Float64Array(0);
  }

  // ========================================================================
  // PRIVATE: INITIALIZATION
  // ========================================================================

  /**
   * Initializes network layers, weights, and buffers
   * Uses He initialization for ReLU activations
   *
   * @formula He init: W ~ N(0, sqrt(2/fan_in))
   */
  private _initNetwork(inputDim: number, outputDim: number): void {
    this._inputDim = inputDim;
    this._outputDim = outputDim;
    this._spatialDim = inputDim;

    // Normalization arrays
    this._inputMean = new Float64Array(inputDim);
    this._inputM2 = new Float64Array(inputDim);
    this._outputMean = new Float64Array(outputDim);
    this._outputM2 = new Float64Array(outputDim);

    // Residual tracking
    this._residualMean = new Float64Array(outputDim);
    this._residualM2 = new Float64Array(outputDim);

    // Temp buffers
    this._tmpNorm = new Float64Array(inputDim);
    this._tmpTarget = new Float64Array(outputDim);
    this._lastInput = new Float64Array(inputDim);
    this._lastOutput = new Float64Array(outputDim);

    // ADWIN window
    this._adwinWindow = new Float64Array(this._maxAdwinSize);

    // Layer dimensions
    this._layerInCh = new Int32Array(this._hiddenLayers);
    this._layerOutCh = new Int32Array(this._hiddenLayers);

    for (let l = 0; l < this._hiddenLayers; l++) {
      this._layerInCh[l] = l === 0 ? 1 : this._convPerLayer;
      this._layerOutCh[l] = this._convPerLayer;
    }

    // Initialize conv layers
    for (let l = 0; l < this._hiddenLayers; l++) {
      const inCh = this._layerInCh[l];
      const outCh = this._layerOutCh[l];
      const kCount = outCh * inCh * this._kernelSize;

      const kernels = new Float64Array(kCount);
      const biases = new Float64Array(outCh);

      // He initialization
      const fanIn = inCh * this._kernelSize;
      const std = Math.sqrt(2.0 / fanIn);

      for (let i = 0; i < kCount; i++) {
        kernels[i] = this._randNormal() * std;
      }
      for (let i = 0; i < outCh; i++) {
        biases[i] = 0.01;
      }

      this._convKernels.push(kernels);
      this._convBiases.push(biases);
      this._convKernelM.push(new Float64Array(kCount));
      this._convKernelV.push(new Float64Array(kCount));
      this._convBiasM.push(new Float64Array(outCh));
      this._convBiasV.push(new Float64Array(outCh));
      this._convKernelGrad.push(new Float64Array(kCount));
      this._convBiasGrad.push(new Float64Array(outCh));

      const actSize = outCh * this._spatialDim;
      this._convPreAct.push(new Float64Array(actSize));
      this._convOutputs.push(new Float64Array(actSize));
      this._convGrads.push(new Float64Array(actSize));

      const inSize = l === 0
        ? this._spatialDim
        : this._convPerLayer * this._spatialDim;
      this._convInputs.push(new Float64Array(inSize));
    }

    // Dense layer
    this._flattenedSize = this._convPerLayer * this._spatialDim;
    const denseCount = this._flattenedSize * outputDim;

    this._denseW = new Float64Array(denseCount);
    this._denseB = new Float64Array(outputDim);
    this._denseWM = new Float64Array(denseCount);
    this._denseWV = new Float64Array(denseCount);
    this._denseBM = new Float64Array(outputDim);
    this._denseBV = new Float64Array(outputDim);
    this._denseWGrad = new Float64Array(denseCount);
    this._denseBGrad = new Float64Array(outputDim);

    const denseStd = Math.sqrt(2.0 / (this._flattenedSize + outputDim));
    for (let i = 0; i < denseCount; i++) {
      this._denseW[i] = this._randNormal() * denseStd;
    }

    this._denseInput = new Float64Array(this._flattenedSize);
    this._denseOutput = new Float64Array(outputDim);
    this._denseGradOut = new Float64Array(outputDim);
    this._denseGradIn = new Float64Array(this._flattenedSize);

    this._initialized = true;
  }

  // ========================================================================
  // PRIVATE: FORWARD PASS
  // ========================================================================

  /**
   * Forward propagation through network
   *
   * @formula Conv1D: y[c,i] = Σₖ Σⱼ(W[c,k,j] · x[k,i+j-pad]) + b[c]
   * @formula ReLU: max(0, x)
   */
  private _forward(input: Float64Array): Float64Array {
    let current = input;

    for (let l = 0; l < this._hiddenLayers; l++) {
      // Cache input
      this._copyTyped(current, this._convInputs[l]);

      // Conv1D
      this._conv1d(
        current,
        this._convKernels[l],
        this._convBiases[l],
        this._convPreAct[l],
        this._layerInCh[l],
        this._layerOutCh[l],
      );

      // ReLU
      const size = this._layerOutCh[l] * this._spatialDim;
      const preAct = this._convPreAct[l];
      const out = this._convOutputs[l];
      for (let i = 0; i < size; i++) {
        out[i] = preAct[i] > 0 ? preAct[i] : 0;
      }

      current = out;
    }

    // Flatten and cache
    this._denseInput.set(current);

    // Dense forward: y = Wx + b
    const flatSize = this._flattenedSize;
    const denseOut = this._denseOutput;
    const denseW = this._denseW;
    const denseB = this._denseB;
    const denseIn = this._denseInput;

    for (let o = 0; o < this._outputDim; o++) {
      let sum = denseB[o];
      const offset = o * flatSize;
      for (let i = 0; i < flatSize; i++) {
        sum += denseW[offset + i] * denseIn[i];
      }
      denseOut[o] = sum;
    }

    return denseOut;
  }

  /**
   * 1D convolution with same padding (optimized)
   */
  private _conv1d(
    input: Float64Array,
    kernel: Float64Array,
    bias: Float64Array,
    output: Float64Array,
    inCh: number,
    outCh: number,
  ): void {
    const spatial = this._spatialDim;
    const ks = this._kernelSize;
    const pad = ks >> 1;

    for (let oc = 0; oc < outCh; oc++) {
      const outOff = oc * spatial;
      const kBase = oc * inCh * ks;
      const b = bias[oc];

      // Initialize with bias
      for (let i = 0; i < spatial; i++) {
        output[outOff + i] = b;
      }

      for (let ic = 0; ic < inCh; ic++) {
        const inOff = ic * spatial;
        const kOff = kBase + ic * ks;

        for (let i = 0; i < spatial; i++) {
          let sum = 0;

          // Unrolled for common kernel size 3
          if (ks === 3) {
            const i0 = i - 1;
            const i1 = i;
            const i2 = i + 1;
            if (i0 >= 0) sum += kernel[kOff] * input[inOff + i0];
            sum += kernel[kOff + 1] * input[inOff + i1];
            if (i2 < spatial) sum += kernel[kOff + 2] * input[inOff + i2];
          } else {
            for (let k = 0; k < ks; k++) {
              const idx = i + k - pad;
              if (idx >= 0 && idx < spatial) {
                sum += kernel[kOff + k] * input[inOff + idx];
              }
            }
          }

          output[outOff + i] += sum;
        }
      }
    }
  }

  // ========================================================================
  // PRIVATE: BACKWARD PASS
  // ========================================================================

  /**
   * Backward propagation with Adam update
   *
   * @formula ∂L/∂W = ∂L/∂y · ∂y/∂W
   * @formula Adam: W -= η·m̂/(√v̂ + ε)
   */
  private _backward(target: Float64Array, lr: number, weight: number): number {
    const flatSize = this._flattenedSize;
    const outDim = this._outputDim;

    // Output gradient: (ŷ - y) / n
    const dOut = this._denseGradOut;
    const pred = this._denseOutput;
    for (let i = 0; i < outDim; i++) {
      dOut[i] = (pred[i] - target[i]) / outDim * weight;
    }

    // Dense gradients
    const dWGrad = this._denseWGrad;
    const dBGrad = this._denseBGrad;
    const dIn = this._denseInput;
    const dW = this._denseW;

    for (let o = 0; o < outDim; o++) {
      const g = dOut[o];
      dBGrad[o] = g;
      const offset = o * flatSize;
      for (let i = 0; i < flatSize; i++) {
        dWGrad[offset + i] = g * dIn[i];
      }
    }

    // Gradient w.r.t. dense input
    const dGradIn = this._denseGradIn;
    for (let i = 0; i < flatSize; i++) {
      let sum = 0;
      for (let o = 0; o < outDim; o++) {
        sum += dW[o * flatSize + i] * dOut[o];
      }
      dGradIn[i] = sum;
    }

    // Add L2 regularization
    for (let i = 0; i < dW.length; i++) {
      dWGrad[i] += this._lambda * dW[i];
    }

    // Adam update for dense
    this._adamUpdate(dW, dWGrad, this._denseWM, this._denseWV, lr);
    this._adamUpdate(this._denseB, dBGrad, this._denseBM, this._denseBV, lr);

    // Backprop through conv layers
    let upstream = dGradIn;

    for (let l = this._hiddenLayers - 1; l >= 0; l--) {
      const outCh = this._layerOutCh[l];
      const inCh = this._layerInCh[l];
      const actSize = outCh * this._spatialDim;

      // ReLU backward
      const preAct = this._convPreAct[l];
      const grad = this._convGrads[l];
      for (let i = 0; i < actSize; i++) {
        grad[i] = preAct[i] > 0 ? upstream[i] : 0;
      }

      // Conv backward
      this._conv1dBackward(
        this._convInputs[l],
        grad,
        this._convKernelGrad[l],
        this._convBiasGrad[l],
        inCh,
        outCh,
      );

      // Add L2
      const kGrad = this._convKernelGrad[l];
      const kW = this._convKernels[l];
      for (let i = 0; i < kW.length; i++) {
        kGrad[i] += this._lambda * kW[i];
      }

      // Adam update
      this._adamUpdate(
        kW,
        kGrad,
        this._convKernelM[l],
        this._convKernelV[l],
        lr,
      );
      this._adamUpdate(
        this._convBiases[l],
        this._convBiasGrad[l],
        this._convBiasM[l],
        this._convBiasV[l],
        lr,
      );

      // Input gradient for next layer
      if (l > 0) {
        this._conv1dInputGrad(grad, kW, this._convInputs[l], inCh, outCh);
        upstream = this._convInputs[l];
      }
    }

    return this._computeGradNorm();
  }

  /**
   * Computes conv kernel and bias gradients
   *
   * @formula ∂L/∂K[oc,ic,k] = Σᵢ ∂L/∂y[oc,i] · x[ic, i+k-pad]
   */
  private _conv1dBackward(
    input: Float64Array,
    gradOut: Float64Array,
    kGrad: Float64Array,
    bGrad: Float64Array,
    inCh: number,
    outCh: number,
  ): void {
    const spatial = this._spatialDim;
    const ks = this._kernelSize;
    const pad = ks >> 1;

    // Zero gradients
    for (let i = 0; i < kGrad.length; i++) kGrad[i] = 0;
    for (let i = 0; i < bGrad.length; i++) bGrad[i] = 0;

    for (let oc = 0; oc < outCh; oc++) {
      const outOff = oc * spatial;
      const kBase = oc * inCh * ks;

      // Bias gradient
      let bSum = 0;
      for (let i = 0; i < spatial; i++) {
        bSum += gradOut[outOff + i];
      }
      bGrad[oc] = bSum;

      // Kernel gradients
      for (let ic = 0; ic < inCh; ic++) {
        const inOff = ic * spatial;
        const kOff = kBase + ic * ks;

        for (let k = 0; k < ks; k++) {
          let kSum = 0;
          for (let i = 0; i < spatial; i++) {
            const idx = i + k - pad;
            if (idx >= 0 && idx < spatial) {
              kSum += gradOut[outOff + i] * input[inOff + idx];
            }
          }
          kGrad[kOff + k] = kSum;
        }
      }
    }
  }

  /**
   * Computes gradient w.r.t. conv input
   *
   * @formula ∂L/∂x[ic,i] = Σₒc Σₖ ∂L/∂y[oc, i-k+pad] · K[oc,ic,k]
   */
  private _conv1dInputGrad(
    gradOut: Float64Array,
    kernel: Float64Array,
    gradIn: Float64Array,
    inCh: number,
    outCh: number,
  ): void {
    const spatial = this._spatialDim;
    const ks = this._kernelSize;
    const pad = ks >> 1;

    for (let i = 0; i < gradIn.length; i++) gradIn[i] = 0;

    for (let ic = 0; ic < inCh; ic++) {
      const inOff = ic * spatial;

      for (let oc = 0; oc < outCh; oc++) {
        const outOff = oc * spatial;
        const kOff = oc * inCh * ks + ic * ks;

        for (let i = 0; i < spatial; i++) {
          let sum = 0;
          for (let k = 0; k < ks; k++) {
            const outIdx = i - k + pad;
            if (outIdx >= 0 && outIdx < spatial) {
              sum += gradOut[outOff + outIdx] * kernel[kOff + k];
            }
          }
          gradIn[inOff + i] += sum;
        }
      }
    }
  }

  /**
   * Adam optimizer update
   *
   * @formula m = β₁·m + (1-β₁)·g
   * @formula v = β₂·v + (1-β₂)·g²
   * @formula W = W - η·(m/(1-β₁ᵗ))/(√(v/(1-β₂ᵗ)) + ε)
   */
  private _adamUpdate(
    w: Float64Array,
    grad: Float64Array,
    m: Float64Array,
    v: Float64Array,
    lr: number,
  ): void {
    const t = this._updateCount;
    const b1Corr = 1 - Math.pow(this._beta1, t);
    const b2Corr = 1 - Math.pow(this._beta2, t);
    const b1 = this._beta1;
    const b2 = this._beta2;
    const eps = this._epsilon;

    const len = w.length;
    for (let i = 0; i < len; i++) {
      const g = grad[i];
      m[i] = b1 * m[i] + (1 - b1) * g;
      v[i] = b2 * v[i] + (1 - b2) * g * g;

      const mHat = m[i] / b1Corr;
      const vHat = v[i] / b2Corr;

      w[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  // ========================================================================
  // PRIVATE: BATCH TRAINING HELPERS
  // ========================================================================

  private _zeroGradients(): void {
    for (let l = 0; l < this._hiddenLayers; l++) {
      const kg = this._convKernelGrad[l];
      const bg = this._convBiasGrad[l];
      for (let i = 0; i < kg.length; i++) kg[i] = 0;
      for (let i = 0; i < bg.length; i++) bg[i] = 0;
    }
    const dwg = this._denseWGrad;
    const dbg = this._denseBGrad;
    for (let i = 0; i < dwg.length; i++) dwg[i] = 0;
    for (let i = 0; i < dbg.length; i++) dbg[i] = 0;
  }

  private _accumulateGrads(target: Float64Array): void {
    const flatSize = this._flattenedSize;
    const outDim = this._outputDim;

    // Output gradient
    const dOut = this._denseGradOut;
    const pred = this._denseOutput;
    for (let i = 0; i < outDim; i++) {
      dOut[i] = (pred[i] - target[i]) / outDim;
    }

    // Accumulate dense gradients
    const dWGrad = this._denseWGrad;
    const dBGrad = this._denseBGrad;
    const dIn = this._denseInput;

    for (let o = 0; o < outDim; o++) {
      const g = dOut[o];
      dBGrad[o] += g;
      const offset = o * flatSize;
      for (let i = 0; i < flatSize; i++) {
        dWGrad[offset + i] += g * dIn[i];
      }
    }

    // Dense input gradient
    const dGradIn = this._denseGradIn;
    const dW = this._denseW;
    for (let i = 0; i < flatSize; i++) {
      let sum = 0;
      for (let o = 0; o < outDim; o++) {
        sum += dW[o * flatSize + i] * dOut[o];
      }
      dGradIn[i] = sum;
    }

    // Conv layers
    let upstream = dGradIn;

    for (let l = this._hiddenLayers - 1; l >= 0; l--) {
      const outCh = this._layerOutCh[l];
      const inCh = this._layerInCh[l];
      const actSize = outCh * this._spatialDim;

      // ReLU backward
      const preAct = this._convPreAct[l];
      const grad = this._convGrads[l];
      for (let i = 0; i < actSize; i++) {
        grad[i] = preAct[i] > 0 ? upstream[i] : 0;
      }

      // Accumulate conv gradients
      this._accumConvGrad(
        this._convInputs[l],
        grad,
        this._convKernelGrad[l],
        this._convBiasGrad[l],
        inCh,
        outCh,
      );

      if (l > 0) {
        this._conv1dInputGrad(
          grad,
          this._convKernels[l],
          this._convInputs[l],
          inCh,
          outCh,
        );
        upstream = this._convInputs[l];
      }
    }
  }

  private _accumConvGrad(
    input: Float64Array,
    gradOut: Float64Array,
    kGrad: Float64Array,
    bGrad: Float64Array,
    inCh: number,
    outCh: number,
  ): void {
    const spatial = this._spatialDim;
    const ks = this._kernelSize;
    const pad = ks >> 1;

    for (let oc = 0; oc < outCh; oc++) {
      const outOff = oc * spatial;
      const kBase = oc * inCh * ks;

      for (let i = 0; i < spatial; i++) {
        bGrad[oc] += gradOut[outOff + i];
      }

      for (let ic = 0; ic < inCh; ic++) {
        const inOff = ic * spatial;
        const kOff = kBase + ic * ks;

        for (let k = 0; k < ks; k++) {
          for (let i = 0; i < spatial; i++) {
            const idx = i + k - pad;
            if (idx >= 0 && idx < spatial) {
              kGrad[kOff + k] += gradOut[outOff + i] * input[inOff + idx];
            }
          }
        }
      }
    }
  }

  private _scaleGradients(scale: number): void {
    for (let l = 0; l < this._hiddenLayers; l++) {
      const kg = this._convKernelGrad[l];
      const bg = this._convBiasGrad[l];
      for (let i = 0; i < kg.length; i++) kg[i] *= scale;
      for (let i = 0; i < bg.length; i++) bg[i] *= scale;
    }
    const dwg = this._denseWGrad;
    const dbg = this._denseBGrad;
    for (let i = 0; i < dwg.length; i++) dwg[i] *= scale;
    for (let i = 0; i < dbg.length; i++) dbg[i] *= scale;
  }

  private _addRegGradients(): void {
    for (let l = 0; l < this._hiddenLayers; l++) {
      const kw = this._convKernels[l];
      const kg = this._convKernelGrad[l];
      for (let i = 0; i < kw.length; i++) {
        kg[i] += this._lambda * kw[i];
      }
    }
    const dw = this._denseW;
    const dwg = this._denseWGrad;
    for (let i = 0; i < dw.length; i++) {
      dwg[i] += this._lambda * dw[i];
    }
  }

  private _applyAdam(lr: number): void {
    for (let l = 0; l < this._hiddenLayers; l++) {
      this._adamUpdate(
        this._convKernels[l],
        this._convKernelGrad[l],
        this._convKernelM[l],
        this._convKernelV[l],
        lr,
      );
      this._adamUpdate(
        this._convBiases[l],
        this._convBiasGrad[l],
        this._convBiasM[l],
        this._convBiasV[l],
        lr,
      );
    }
    this._adamUpdate(
      this._denseW,
      this._denseWGrad,
      this._denseWM,
      this._denseWV,
      lr,
    );
    this._adamUpdate(
      this._denseB,
      this._denseBGrad,
      this._denseBM,
      this._denseBV,
      lr,
    );
  }

  // ========================================================================
  // PRIVATE: NORMALIZATION (Welford's Algorithm)
  // ========================================================================

  /**
   * Updates running statistics using Welford's online algorithm
   *
   * @formula δ = x - μ, μ += δ/n, M₂ += δ·(x - μ)
   */
  private _updateNormStats(
    x: number[] | Float64Array,
    y: number[] | Float64Array,
  ): void {
    this._normCount++;
    const n = this._normCount;

    for (let i = 0; i < this._inputDim; i++) {
      const delta = x[i] - this._inputMean[i];
      this._inputMean[i] += delta / n;
      const delta2 = x[i] - this._inputMean[i];
      this._inputM2[i] += delta * delta2;
    }

    for (let i = 0; i < this._outputDim; i++) {
      const delta = y[i] - this._outputMean[i];
      this._outputMean[i] += delta / n;
      const delta2 = y[i] - this._outputMean[i];
      this._outputM2[i] += delta * delta2;
    }
  }

  private _normalizeInput(x: number[] | Float64Array, out: Float64Array): void {
    for (let i = 0; i < this._inputDim; i++) {
      out[i] = (x[i] - this._inputMean[i]) / this._getInputStd(i);
    }
  }

  private _normalizeOutput(
    y: number[] | Float64Array,
    out: Float64Array,
  ): void {
    for (let i = 0; i < this._outputDim; i++) {
      out[i] = (y[i] - this._outputMean[i]) / this._getOutputStd(i);
    }
  }

  private _getInputStd(i: number): number {
    if (this._normCount < 2) return 1;
    const variance = this._inputM2[i] / (this._normCount - 1);
    return Math.sqrt(Math.max(variance, this._epsilon)) + this._epsilon;
  }

  private _getOutputStd(i: number): number {
    if (this._normCount < 2) return 1;
    const variance = this._outputM2[i] / (this._normCount - 1);
    return Math.sqrt(Math.max(variance, this._epsilon)) + this._epsilon;
  }

  // ========================================================================
  // PRIVATE: LEARNING RATE SCHEDULE
  // ========================================================================

  /**
   * Computes LR with linear warmup and cosine decay
   *
   * @formula Warmup: η = η_base · t/warmup
   * @formula Cosine: η = η_base · 0.5 · (1 + cos(π·progress))
   */
  private _computeLR(): number {
    const t = this._updateCount;

    if (t < this._warmupSteps) {
      return this._baseLR * (t + 1) / this._warmupSteps;
    }

    const progress = Math.min(
      1,
      (t - this._warmupSteps) /
        Math.max(1, this._totalSteps - this._warmupSteps),
    );
    return this._baseLR * 0.5 * (1 + Math.cos(Math.PI * progress));
  }

  // ========================================================================
  // PRIVATE: REGULARIZATION AND METRICS
  // ========================================================================

  /**
   * @formula L_reg = (λ/2)·Σ||W||²
   */
  private _computeRegLoss(): number {
    let loss = 0;

    for (let l = 0; l < this._hiddenLayers; l++) {
      const k = this._convKernels[l];
      for (let i = 0; i < k.length; i++) {
        loss += k[i] * k[i];
      }
    }

    const dw = this._denseW;
    for (let i = 0; i < dw.length; i++) {
      loss += dw[i] * dw[i];
    }

    return 0.5 * this._lambda * loss;
  }

  private _computeGradNorm(): number {
    let norm = 0;

    for (let l = 0; l < this._hiddenLayers; l++) {
      const kg = this._convKernelGrad[l];
      const bg = this._convBiasGrad[l];
      for (let i = 0; i < kg.length; i++) norm += kg[i] * kg[i];
      for (let i = 0; i < bg.length; i++) norm += bg[i] * bg[i];
    }

    const dwg = this._denseWGrad;
    const dbg = this._denseBGrad;
    for (let i = 0; i < dwg.length; i++) norm += dwg[i] * dwg[i];
    for (let i = 0; i < dbg.length; i++) norm += dbg[i] * dbg[i];

    return Math.sqrt(norm);
  }

  private _computeAccuracy(): number {
    if (this._sampleCount === 0) return 0;
    const avgLoss = this._totalLoss / this._sampleCount;
    return 1 / (1 + avgLoss);
  }

  private _countParams(): number {
    if (!this._initialized) return 0;

    let total = 0;
    for (let l = 0; l < this._hiddenLayers; l++) {
      total += this._convKernels[l].length + this._convBiases[l].length;
    }
    total += this._denseW.length + this._denseB.length;

    return total;
  }

  // ========================================================================
  // PRIVATE: OUTLIER DETECTION
  // ========================================================================

  /**
   * @formula r = (y - ŷ)/σ, outlier if |r| > threshold
   */
  private _detectOutlier(pred: Float64Array, target: Float64Array): boolean {
    if (this._residualCount < 10) return false;

    for (let i = 0; i < this._outputDim; i++) {
      const residual = target[i] - pred[i];
      const variance = this._residualM2[i] / (this._residualCount - 1);
      const std = Math.sqrt(Math.max(variance, this._epsilon));
      const z = Math.abs(residual - this._residualMean[i]) /
        (std + this._epsilon);

      if (z > this._outlierThreshold) return true;
    }

    return false;
  }

  private _updateResidualStats(pred: Float64Array, target: Float64Array): void {
    this._residualCount++;
    const n = this._residualCount;

    for (let i = 0; i < this._outputDim; i++) {
      const r = target[i] - pred[i];
      const delta = r - this._residualMean[i];
      this._residualMean[i] += delta / n;
      const delta2 = r - this._residualMean[i];
      this._residualM2[i] += delta * delta2;
    }
  }

  // ========================================================================
  // PRIVATE: ADWIN DRIFT DETECTION
  // ========================================================================

  /**
   * ADWIN drift detection using sliding window
   *
   * @formula εcut = sqrt((1/2m)·ln(4/δ))
   * @formula Drift if |μ₀ - μ₁| ≥ εcut
   */
  private _updateAdwin(error: number): boolean {
    const win = this._adwinWindow;
    const maxSize = this._maxAdwinSize;

    // Add to circular buffer
    win[this._adwinHead] = error;
    this._adwinHead = (this._adwinHead + 1) % maxSize;
    this._adwinSize = Math.min(this._adwinSize + 1, maxSize);

    this._adwinSum += error;
    this._adwinSumSq += error * error;

    if (this._adwinSize < 30) return false;

    // Check for drift by comparing window halves
    const halfSize = this._adwinSize >> 1;

    let sum1 = 0, sum2 = 0;
    const startIdx = (this._adwinHead - this._adwinSize + maxSize) % maxSize;

    for (let i = 0; i < halfSize; i++) {
      sum1 += win[(startIdx + i) % maxSize];
    }
    for (let i = halfSize; i < this._adwinSize; i++) {
      sum2 += win[(startIdx + i) % maxSize];
    }

    const mean1 = sum1 / halfSize;
    const mean2 = sum2 / (this._adwinSize - halfSize);

    const m = Math.min(halfSize, this._adwinSize - halfSize);
    const epsilon = Math.sqrt((1 / (2 * m)) * Math.log(4 / this._adwinDelta));

    if (Math.abs(mean1 - mean2) > epsilon) {
      this._driftCount++;

      // Keep only recent half
      this._adwinSize = this._adwinSize - halfSize;
      this._adwinSum = sum2;

      // Decay normalization stats
      const decay = 0.9;
      for (let i = 0; i < this._inputDim; i++) {
        this._inputM2[i] *= decay;
      }
      for (let i = 0; i < this._outputDim; i++) {
        this._outputM2[i] *= decay;
      }

      return true;
    }

    return false;
  }

  // ========================================================================
  // PRIVATE: UTILITIES
  // ========================================================================

  /**
   * Box-Muller transform for normal distribution
   */
  private _randNormal(): number {
    if (this._hasSpare) {
      this._hasSpare = false;
      return this._spare;
    }

    const u = Math.random();
    const v = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u + 1e-10));

    this._spare = mag * Math.sin(2 * Math.PI * v);
    this._hasSpare = true;

    return mag * Math.cos(2 * Math.PI * v);
  }

  private _copyTyped(src: Float64Array, dst: Float64Array): void {
    dst.set(src.subarray(0, Math.min(src.length, dst.length)));
  }

  private _copyTo(src: number[] | Float64Array, dst: Float64Array): void {
    const len = Math.min(src.length, dst.length);
    if (src instanceof Float64Array) {
      dst.set(src.subarray(0, len));
    } else {
      for (let i = 0; i < len; i++) dst[i] = src[i];
    }
  }

  private _emptyFitResult(): FitResult {
    return {
      loss: 0,
      gradientNorm: 0,
      effectiveLearningRate: 0,
      isOutlier: false,
      converged: false,
      sampleIndex: this._sampleCount,
      driftDetected: false,
    };
  }
}
