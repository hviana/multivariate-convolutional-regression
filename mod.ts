/**
 * TCNRegression: Temporal Convolutional Network for Multivariate Regression
 *
 * A causal dilated 1D CNN with incremental online learning, Adam optimizer,
 * and Welford z-score normalization. Designed for tight CPU/memory environments
 * with zero hot-path allocations.
 *
 * @module TCNRegression
 */

// ============================================================================
// TYPE DEFINITIONS AND INTERFACES
// ============================================================================

/**
 * Configuration for the TCN regression model
 */
export interface TCNRegressionConfig {
  /** Maximum lookback window (receptive field cap). Default: 64 */
  maxSequenceLength?: number;
  /** Maximum prediction horizon. Default: 1 */
  maxFutureSteps?: number;
  /** Channels in TCN blocks. Default: 32 */
  hiddenChannels?: number;
  /** Number of residual TCN blocks. Default: 4 */
  nBlocks?: number;
  /** Convolution kernel size. Default: 3 */
  kernelSize?: number;
  /** Dilation growth factor (dilations = base^blockIndex). Default: 2 */
  dilationBase?: number;
  /** Use 2 conv layers per TCN block. Default: true */
  useTwoLayerBlock?: boolean;
  /** Activation function. Default: "relu" */
  activation?: "relu" | "gelu";
  /** Enable channel normalization. Default: false */
  useLayerNorm?: boolean;
  /** Dropout probability (training only). Default: 0.0 */
  dropoutRate?: number;
  /** Learning rate. Default: 0.001 */
  learningRate?: number;
  /** Adam first moment decay. Default: 0.9 */
  beta1?: number;
  /** Adam second moment decay. Default: 0.999 */
  beta2?: number;
  /** Adam epsilon for stability. Default: 1e-8 */
  epsilon?: number;
  /** L2 regularization coefficient. Default: 0.0001 */
  l2Lambda?: number;
  /** Max gradient L2 norm. Default: 1.0 */
  gradientClipNorm?: number;
  /** Welford variance floor. Default: 1e-8 */
  normalizationEpsilon?: number;
  /** Samples before applying z-score. Default: 10 */
  normalizationWarmup?: number;
  /** Z-score threshold for downweighting. Default: 3.0 */
  outlierThreshold?: number;
  /** Minimum sample weight for outliers. Default: 0.1 */
  outlierMinWeight?: number;
  /** Enable ADWIN drift detection. Default: true */
  adwinEnabled?: boolean;
  /** ADWIN significance parameter. Default: 0.002 */
  adwinDelta?: number;
  /** Max ADWIN bucket count (memory cap). Default: 64 */
  adwinMaxBuckets?: number;
  /** Direct head vs recursive rollforward. Default: true */
  useDirectMultiHorizon?: boolean;
  /** Samples for uncertainty estimation. Default: 100 */
  residualWindowSize?: number;
  /** Z-multiplier for confidence bounds. Default: 1.96 */
  uncertaintyMultiplier?: number;
  /** Xavier/He init scale factor. Default: 0.1 */
  weightInitScale?: number;
  /** Deterministic RNG seed. Default: 42 */
  seed?: number;
  /** Enable debug logging. Default: false */
  verbose?: boolean;
}

/** Result returned from fitOnline */
export interface FitResult {
  loss: number;
  sampleWeight: number;
  driftDetected: boolean;
  metrics: {
    avgLoss: number;
    mae: number;
    sampleCount: number;
  };
}

/** Result returned from predict */
export interface PredictionResult {
  /** Predictions array [futureSteps][nTargets] */
  predictions: number[][];
  /** Lower uncertainty bounds [futureSteps][nTargets] */
  uncertaintyLower: number[][];
  /** Upper uncertainty bounds [futureSteps][nTargets] */
  uncertaintyUpper: number[][];
  /** Confidence score 0-1 */
  confidence: number;
}

/** Model architecture summary */
export interface ModelSummary {
  architecture: string;
  totalParameters: number;
  layerParameters: { [key: string]: number };
  receptiveField: number;
  memoryUsageBytes: number;
  config: Required<TCNRegressionConfig>;
}

/** Weight information for all parameters */
export interface WeightInfo {
  [layerName: string]: {
    weights: number[];
    shape: number[];
    bias?: number[];
  };
}

/** Normalization statistics */
export interface NormalizationStats {
  inputMeans: number[];
  inputStds: number[];
  outputMeans: number[];
  outputStds: number[];
  sampleCount: number;
  warmupComplete: boolean;
}

// ============================================================================
// PHASE 1: MEMORY INFRASTRUCTURE
// ============================================================================

/**
 * Immutable descriptor holding dimensions and precomputed strides for row-major layout
 */
class TensorShape {
  readonly dims: readonly number[];
  readonly strides: readonly number[];
  readonly numel: number;

  /**
   * @param dims - Array of dimension sizes (must be positive integers)
   * @throws Error if any dimension is not a positive integer
   */
  constructor(dims: number[]) {
    for (let i = 0; i < dims.length; i++) {
      if (!Number.isInteger(dims[i]) || dims[i] <= 0) {
        throw new Error(`Invalid dimension at index ${i}: ${dims[i]}`);
      }
    }
    this.dims = Object.freeze([...dims]);

    // Compute strides for row-major layout: stride[i] = product(dims[i+1:])
    const strides = new Array(dims.length);
    let stride = 1;
    for (let i = dims.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= dims[i];
    }
    this.strides = Object.freeze(strides);
    this.numel = stride;
  }

  /**
   * Compute flat offset from multi-dimensional indices
   * @param indices - Array of indices, one per dimension
   * @returns Flat offset into data array
   */
  index(...indices: number[]): number {
    let offset = 0;
    for (let i = 0; i < indices.length; i++) {
      offset += indices[i] * this.strides[i];
    }
    return offset;
  }

  get rank(): number {
    return this.dims.length;
  }
}

/**
 * Zero-copy view into a Float64Array slab
 */
class TensorView {
  readonly data: Float64Array;
  readonly offset: number;
  readonly shape: TensorShape;

  constructor(data: Float64Array, offset: number, shape: TensorShape) {
    if (offset + shape.numel > data.length) {
      throw new Error(
        `TensorView out of bounds: offset=${offset}, numel=${shape.numel}, dataLen=${data.length}`,
      );
    }
    this.data = data;
    this.offset = offset;
    this.shape = shape;
  }

  get(...indices: number[]): number {
    return this.data[this.offset + this.shape.index(...indices)];
  }

  set(value: number, ...indices: number[]): void {
    this.data[this.offset + this.shape.index(...indices)] = value;
  }

  fill(value: number): void {
    const end = this.offset + this.shape.numel;
    for (let i = this.offset; i < end; i++) {
      this.data[i] = value;
    }
  }

  copyFrom(other: TensorView): void {
    const n = this.shape.numel;
    for (let i = 0; i < n; i++) {
      this.data[this.offset + i] = other.data[other.offset + i];
    }
  }

  /** Get underlying flat array segment */
  toArray(): number[] {
    const result = new Array(this.shape.numel);
    for (let i = 0; i < this.shape.numel; i++) {
      result[i] = this.data[this.offset + i];
    }
    return result;
  }
}

/**
 * Manages reusable scratch buffers organized by size class (powers of two)
 */
class BufferPool {
  private pools: Map<number, Float64Array[]> = new Map();
  private highWaterMark: Map<number, number> = new Map();
  private static SIZE_CLASSES = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
  ];

  private getSizeClass(minSize: number): number {
    for (const sc of BufferPool.SIZE_CLASSES) {
      if (sc >= minSize) return sc;
    }
    // Round up to next power of 2 for very large buffers
    return Math.pow(2, Math.ceil(Math.log2(minSize)));
  }

  /**
   * Rent a buffer of at least minSize elements
   * @param minSize - Minimum number of elements needed
   * @returns Buffer from pool or newly created
   */
  rent(minSize: number): Float64Array {
    const sizeClass = this.getSizeClass(minSize);
    let pool = this.pools.get(sizeClass);
    if (!pool) {
      pool = [];
      this.pools.set(sizeClass, pool);
    }

    if (pool.length > 0) {
      return pool.pop()!;
    }

    // Track high water mark
    const current = this.highWaterMark.get(sizeClass) || 0;
    this.highWaterMark.set(sizeClass, current + 1);

    return new Float64Array(sizeClass);
  }

  /**
   * Return a buffer to the pool
   * @param buffer - Buffer to recycle
   */
  return(buffer: Float64Array): void {
    const sizeClass = buffer.length;
    let pool = this.pools.get(sizeClass);
    if (!pool) {
      pool = [];
      this.pools.set(sizeClass, pool);
    }
    // Zero buffer before returning for safety
    buffer.fill(0);
    pool.push(buffer);
  }

  getStats(): { sizeClass: number; pooled: number; allocated: number }[] {
    const stats: { sizeClass: number; pooled: number; allocated: number }[] =
      [];
    for (const [sizeClass, pool] of this.pools) {
      stats.push({
        sizeClass,
        pooled: pool.length,
        allocated: this.highWaterMark.get(sizeClass) || 0,
      });
    }
    return stats;
  }
}

/**
 * Preallocated contiguous Float64Array slab with bump allocation
 */
class TensorArena {
  private data: Float64Array;
  private offset: number = 0;

  constructor(totalSize: number) {
    this.data = new Float64Array(totalSize);
  }

  /**
   * Allocate a tensor view of given shape
   * @param shape - Shape of tensor to allocate
   * @returns TensorView at current offset
   * @throws Error if allocation exceeds capacity
   */
  alloc(shape: TensorShape): TensorView {
    if (this.offset + shape.numel > this.data.length) {
      throw new Error(
        `TensorArena overflow: need ${shape.numel}, have ${
          this.data.length - this.offset
        }`,
      );
    }
    const view = new TensorView(this.data, this.offset, shape);
    this.offset += shape.numel;
    return view;
  }

  /** Return current allocation offset for mark/release */
  mark(): number {
    return this.offset;
  }

  /** Release allocations back to mark point */
  release(mark: number): void {
    this.offset = mark;
  }

  /** Reset arena to empty state */
  reset(): void {
    this.offset = 0;
  }

  get usedBytes(): number {
    return this.offset * 8;
  }

  get totalBytes(): number {
    return this.data.length * 8;
  }
}

/**
 * Static utility class with low-level tensor operations
 * All operations use explicit offsets/strides with no allocations
 */
class TensorOps {
  static fill(
    data: Float64Array,
    offset: number,
    len: number,
    value: number,
  ): void {
    const end = offset + len;
    for (let i = offset; i < end; i++) {
      data[i] = value;
    }
  }

  static copy(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOff + i] = src[srcOff + i];
    }
  }

  static add(
    dst: Float64Array,
    dstOff: number,
    a: Float64Array,
    aOff: number,
    b: Float64Array,
    bOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOff + i] = a[aOff + i] + b[bOff + i];
    }
  }

  static addInPlace(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOff + i] += src[srcOff + i];
    }
  }

  static scale(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    s: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOff + i] = src[srcOff + i] * s;
    }
  }

  static scaleInPlace(
    dst: Float64Array,
    dstOff: number,
    s: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOff + i] *= s;
    }
  }

  /** dst = a * x + y (axpy operation) */
  static axpy(
    dst: Float64Array,
    dstOff: number,
    a: number,
    x: Float64Array,
    xOff: number,
    y: Float64Array,
    yOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOff + i] = a * x[xOff + i] + y[yOff + i];
    }
  }

  static dot(
    a: Float64Array,
    aOff: number,
    b: Float64Array,
    bOff: number,
    len: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += a[aOff + i] * b[bOff + i];
    }
    return sum;
  }

  static sum(data: Float64Array, offset: number, len: number): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += data[offset + i];
    }
    return sum;
  }

  static max(data: Float64Array, offset: number, len: number): number {
    let max = data[offset];
    for (let i = 1; i < len; i++) {
      if (data[offset + i] > max) max = data[offset + i];
    }
    return max;
  }

  /** Matrix multiply: dst = A @ B where A is [M,K], B is [K,N], dst is [M,N] */
  static matmul(
    dst: Float64Array,
    dstOff: number,
    A: Float64Array,
    aOff: number,
    B: Float64Array,
    bOff: number,
    M: number,
    K: number,
    N: number,
  ): void {
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += A[aOff + i * K + k] * B[bOff + k * N + j];
        }
        dst[dstOff + i * N + j] = sum;
      }
    }
  }

  /** Transpose: dst[j,i] = src[i,j] */
  static transpose(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    rows: number,
    cols: number,
  ): void {
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        dst[dstOff + j * rows + i] = src[srcOff + i * cols + j];
      }
    }
  }

  /** Compute L2 norm */
  static l2Norm(data: Float64Array, offset: number, len: number): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const v = data[offset + i];
      sum += v * v;
    }
    return Math.sqrt(sum);
  }
}

// ============================================================================
// PHASE 2: NUMERICAL UTILITIES
// ============================================================================

/**
 * Deterministic PRNG using xorshift128+ algorithm
 */
class RandomGenerator {
  private s0: number;
  private s1: number;
  private cachedGaussian: number | null = null;

  constructor(seed: number = 42) {
    // Initialize state from seed using splitmix64
    this.s0 = this.splitmix64(seed);
    this.s1 = this.splitmix64(this.s0);
  }

  private splitmix64(x: number): number {
    x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
    return (x ^ (x >>> 16)) >>> 0;
  }

  /** Returns uniform random number in [0, 1) */
  nextFloat(): number {
    // xorshift128+
    let s1 = this.s0;
    const s0 = this.s1;
    this.s0 = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >>> 17;
    s1 ^= s0;
    s1 ^= s0 >>> 26;
    this.s1 = s1;
    return ((this.s0 + this.s1) >>> 0) / 4294967296;
  }

  /** Returns Gaussian random number using Box-Muller transform */
  nextGaussian(): number {
    if (this.cachedGaussian !== null) {
      const result = this.cachedGaussian;
      this.cachedGaussian = null;
      return result;
    }

    let u1: number, u2: number;
    do {
      u1 = this.nextFloat();
      u2 = this.nextFloat();
    } while (u1 <= 1e-10);

    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const z0 = mag * Math.cos(2.0 * Math.PI * u2);
    const z1 = mag * Math.sin(2.0 * Math.PI * u2);

    this.cachedGaussian = z1;
    return z0;
  }

  /** Returns truncated Gaussian (rejects samples outside limit) */
  truncatedGaussian(std: number, limit: number): number {
    let sample: number;
    do {
      sample = this.nextGaussian() * std;
    } while (Math.abs(sample) > limit);
    return sample;
  }
}

/**
 * Static class for activation functions and derivatives
 * Uses precomputed constants for GELU approximation
 */
class ActivationOps {
  // GELU constants: sqrt(2/pi) ≈ 0.7978845608
  private static readonly GELU_COEF = 0.044715;
  private static readonly SQRT_2_PI = 0.7978845608028654;

  static relu(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOff + i] = Math.max(0, src[srcOff + i]);
    }
  }

  static reluInPlace(data: Float64Array, offset: number, len: number): void {
    for (let i = 0; i < len; i++) {
      if (data[offset + i] < 0) data[offset + i] = 0;
    }
  }

  static reluBackward(
    dOut: Float64Array,
    dOutOff: number,
    dIn: Float64Array,
    dInOff: number,
    preAct: Float64Array,
    preActOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dIn[dInOff + i] = preAct[preActOff + i] > 0 ? dOut[dOutOff + i] : 0;
    }
  }

  /** GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
  static gelu(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      const x = src[srcOff + i];
      const x3 = x * x * x;
      const inner = this.SQRT_2_PI * (x + this.GELU_COEF * x3);
      dst[dstOff + i] = 0.5 * x * (1 + Math.tanh(inner));
    }
  }

  static geluInPlace(data: Float64Array, offset: number, len: number): void {
    for (let i = 0; i < len; i++) {
      const x = data[offset + i];
      const x3 = x * x * x;
      const inner = this.SQRT_2_PI * (x + this.GELU_COEF * x3);
      data[offset + i] = 0.5 * x * (1 + Math.tanh(inner));
    }
  }

  /** GELU derivative approximation */
  static geluBackward(
    dOut: Float64Array,
    dOutOff: number,
    dIn: Float64Array,
    dInOff: number,
    preAct: Float64Array,
    preActOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      const x = preAct[preActOff + i];
      const x2 = x * x;
      const x3 = x2 * x;
      const inner = this.SQRT_2_PI * (x + this.GELU_COEF * x3);
      const tanh_inner = Math.tanh(inner);
      const sech2 = 1 - tanh_inner * tanh_inner;
      const dInner = this.SQRT_2_PI * (1 + 3 * this.GELU_COEF * x2);
      const dGelu = 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * dInner;
      dIn[dInOff + i] = dOut[dOutOff + i] * dGelu;
    }
  }
}

/**
 * Tracks running mean and variance using Welford online algorithm
 * Numerically stable for streaming data
 */
class WelfordAccumulator {
  private count: number = 0;
  private mean: number = 0;
  private m2: number = 0; // Sum of squared deviations

  /**
   * Update with a new value using Welford's algorithm
   * @param value - New observation
   */
  update(value: number): void {
    this.count++;
    const delta = value - this.mean;
    this.mean += delta / this.count;
    const delta2 = value - this.mean;
    this.m2 += delta * delta2;
  }

  getMean(): number {
    return this.mean;
  }

  getVariance(epsilon: number = 1e-8): number {
    if (this.count < 2) return epsilon;
    return Math.max(this.m2 / (this.count - 1), epsilon);
  }

  getStd(epsilon: number = 1e-8): number {
    return Math.sqrt(this.getVariance(epsilon));
  }

  getCount(): number {
    return this.count;
  }

  reset(): void {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }

  serialize(): { count: number; mean: number; m2: number } {
    return { count: this.count, mean: this.mean, m2: this.m2 };
  }

  deserialize(data: { count: number; mean: number; m2: number }): void {
    this.count = data.count;
    this.mean = data.mean;
    this.m2 = data.m2;
  }
}

/**
 * Manages array of WelfordAccumulator instances for multi-feature normalization
 */
class WelfordNormalizer {
  private inputAccumulators: WelfordAccumulator[];
  private outputAccumulators: WelfordAccumulator[];
  private epsilon: number;
  private warmupSamples: number;
  private nInputFeatures: number;
  private nOutputFeatures: number;

  constructor(
    nInputFeatures: number,
    nOutputFeatures: number,
    epsilon: number = 1e-8,
    warmupSamples: number = 10,
  ) {
    this.nInputFeatures = nInputFeatures;
    this.nOutputFeatures = nOutputFeatures;
    this.epsilon = epsilon;
    this.warmupSamples = warmupSamples;

    this.inputAccumulators = new Array(nInputFeatures);
    for (let i = 0; i < nInputFeatures; i++) {
      this.inputAccumulators[i] = new WelfordAccumulator();
    }

    this.outputAccumulators = new Array(nOutputFeatures);
    for (let i = 0; i < nOutputFeatures; i++) {
      this.outputAccumulators[i] = new WelfordAccumulator();
    }
  }

  updateInputStats(features: number[]): void {
    for (let i = 0; i < features.length && i < this.nInputFeatures; i++) {
      this.inputAccumulators[i].update(features[i]);
    }
  }

  updateOutputStats(targets: number[]): void {
    for (let i = 0; i < targets.length && i < this.nOutputFeatures; i++) {
      this.outputAccumulators[i].update(targets[i]);
    }
  }

  /** Apply z-score normalization to inputs */
  normalizeInputs(dst: Float64Array, dstOff: number, src: number[]): void {
    const count = this.inputAccumulators[0].getCount();
    if (count < this.warmupSamples) {
      // During warmup, just copy
      for (let i = 0; i < src.length && i < this.nInputFeatures; i++) {
        dst[dstOff + i] = src[i];
      }
    } else {
      for (let i = 0; i < src.length && i < this.nInputFeatures; i++) {
        const mean = this.inputAccumulators[i].getMean();
        const std = this.inputAccumulators[i].getStd(this.epsilon);
        dst[dstOff + i] = (src[i] - mean) / std;
      }
    }
  }

  /** Apply z-score normalization to outputs */
  normalizeOutputs(dst: Float64Array, dstOff: number, src: number[]): void {
    const count = this.outputAccumulators[0].getCount();
    if (count < this.warmupSamples) {
      for (let i = 0; i < src.length && i < this.nOutputFeatures; i++) {
        dst[dstOff + i] = src[i];
      }
    } else {
      for (let i = 0; i < src.length && i < this.nOutputFeatures; i++) {
        const mean = this.outputAccumulators[i].getMean();
        const std = this.outputAccumulators[i].getStd(this.epsilon);
        dst[dstOff + i] = (src[i] - mean) / std;
      }
    }
  }

  /** Inverse z-score transform for outputs */
  denormalizeOutputs(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    len: number,
  ): void {
    const count = this.outputAccumulators[0].getCount();
    if (count < this.warmupSamples) {
      for (let i = 0; i < len && i < this.nOutputFeatures; i++) {
        dst[dstOff + i] = src[srcOff + i];
      }
    } else {
      for (let i = 0; i < len && i < this.nOutputFeatures; i++) {
        const mean = this.outputAccumulators[i].getMean();
        const std = this.outputAccumulators[i].getStd(this.epsilon);
        dst[dstOff + i] = src[srcOff + i] * std + mean;
      }
    }
  }

  getStats(): NormalizationStats {
    const inputMeans = this.inputAccumulators.map((a) => a.getMean());
    const inputStds = this.inputAccumulators.map((a) => a.getStd(this.epsilon));
    const outputMeans = this.outputAccumulators.map((a) => a.getMean());
    const outputStds = this.outputAccumulators.map((a) =>
      a.getStd(this.epsilon)
    );
    const sampleCount = this.inputAccumulators[0].getCount();

    return {
      inputMeans,
      inputStds,
      outputMeans,
      outputStds,
      sampleCount,
      warmupComplete: sampleCount >= this.warmupSamples,
    };
  }

  reset(): void {
    for (const acc of this.inputAccumulators) acc.reset();
    for (const acc of this.outputAccumulators) acc.reset();
  }

  serialize(): object {
    return {
      inputAccumulators: this.inputAccumulators.map((a) => a.serialize()),
      outputAccumulators: this.outputAccumulators.map((a) => a.serialize()),
    };
  }

  deserialize(data: any): void {
    for (let i = 0; i < this.inputAccumulators.length; i++) {
      if (data.inputAccumulators[i]) {
        this.inputAccumulators[i].deserialize(data.inputAccumulators[i]);
      }
    }
    for (let i = 0; i < this.outputAccumulators.length; i++) {
      if (data.outputAccumulators[i]) {
        this.outputAccumulators[i].deserialize(data.outputAccumulators[i]);
      }
    }
  }
}

/**
 * Static methods for loss computation
 */
class LossFunction {
  /** Mean Squared Error */
  static mse(
    predictions: Float64Array,
    predOff: number,
    targets: Float64Array,
    targOff: number,
    len: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const diff = predictions[predOff + i] - targets[targOff + i];
      sum += diff * diff;
    }
    return sum / len;
  }

  /** Weighted MSE */
  static mseWeighted(
    predictions: Float64Array,
    predOff: number,
    targets: Float64Array,
    targOff: number,
    weight: number,
    len: number,
  ): number {
    return weight * this.mse(predictions, predOff, targets, targOff, len);
  }

  /** MSE gradient: d/d(pred) = 2 * (pred - target) / n */
  static mseGradient(
    dLoss: Float64Array,
    dLossOff: number,
    predictions: Float64Array,
    predOff: number,
    targets: Float64Array,
    targOff: number,
    len: number,
  ): void {
    const scale = 2.0 / len;
    for (let i = 0; i < len; i++) {
      dLoss[dLossOff + i] = scale *
        (predictions[predOff + i] - targets[targOff + i]);
    }
  }

  /** Weighted MSE gradient */
  static mseGradientWeighted(
    dLoss: Float64Array,
    dLossOff: number,
    predictions: Float64Array,
    predOff: number,
    targets: Float64Array,
    targOff: number,
    weight: number,
    len: number,
  ): void {
    const scale = 2.0 * weight / len;
    for (let i = 0; i < len; i++) {
      dLoss[dLossOff + i] = scale *
        (predictions[predOff + i] - targets[targOff + i]);
    }
  }

  /** L2 regularization term: lambda * sum(param^2) */
  static l2Regularization(
    params: Float64Array,
    offset: number,
    len: number,
    lambda: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const p = params[offset + i];
      sum += p * p;
    }
    return lambda * sum;
  }

  /** L2 gradient: 2 * lambda * param */
  static l2Gradient(
    dParams: Float64Array,
    dOff: number,
    params: Float64Array,
    pOff: number,
    lambda: number,
    len: number,
  ): void {
    const scale = 2 * lambda;
    for (let i = 0; i < len; i++) {
      dParams[dOff + i] += scale * params[pOff + i];
    }
  }
}

// ============================================================================
// PHASE 3: OPTIMIZER
// ============================================================================

/**
 * Holds gradient buffer for a single parameter
 */
class GradientAccumulator {
  readonly gradients: Float64Array;
  readonly size: number;

  constructor(size: number) {
    this.size = size;
    this.gradients = new Float64Array(size);
  }

  accumulate(grads: Float64Array, srcOff: number, len: number): void {
    for (let i = 0; i < len && i < this.size; i++) {
      this.gradients[i] += grads[srcOff + i];
    }
  }

  scale(s: number): void {
    for (let i = 0; i < this.size; i++) {
      this.gradients[i] *= s;
    }
  }

  /** Clip gradients by L2 norm */
  clipByNorm(maxNorm: number): number {
    const norm = TensorOps.l2Norm(this.gradients, 0, this.size);
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      this.scale(scale);
    }
    return norm;
  }

  zero(): void {
    this.gradients.fill(0);
  }
}

/**
 * Adam optimizer state for a single parameter
 * Stores first moment (m) and second moment (v) estimates
 */
class AdamState {
  readonly m: Float64Array; // First moment
  readonly v: Float64Array; // Second moment
  readonly size: number;

  constructor(size: number) {
    this.size = size;
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
  }

  /**
   * Adam update rule with bias correction
   * params[i] -= lr * mHat / (sqrt(vHat) + eps)
   */
  update(
    params: Float64Array,
    pOff: number,
    grads: Float64Array,
    gOff: number,
    lr: number,
    beta1: number,
    beta2: number,
    epsilon: number,
    t: number,
  ): void {
    const beta1Pow = Math.pow(beta1, t);
    const beta2Pow = Math.pow(beta2, t);
    const biasCorrection1 = 1 - beta1Pow;
    const biasCorrection2 = 1 - beta2Pow;

    for (let i = 0; i < this.size; i++) {
      const g = grads[gOff + i];
      // Update biased first moment
      this.m[i] = beta1 * this.m[i] + (1 - beta1) * g;
      // Update biased second moment
      this.v[i] = beta2 * this.v[i] + (1 - beta2) * g * g;
      // Bias-corrected estimates
      const mHat = this.m[i] / biasCorrection1;
      const vHat = this.v[i] / biasCorrection2;
      // Update parameters
      params[pOff + i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
    }
  }

  reset(): void {
    this.m.fill(0);
    this.v.fill(0);
  }

  serialize(): { m: number[]; v: number[] } {
    return {
      m: Array.from(this.m),
      v: Array.from(this.v),
    };
  }

  deserialize(data: { m: number[]; v: number[] }): void {
    for (let i = 0; i < this.size; i++) {
      this.m[i] = data.m[i] || 0;
      this.v[i] = data.v[i] || 0;
    }
  }
}

/**
 * Adam optimizer managing all parameters
 */
class AdamOptimizer {
  private lr: number;
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private clipNorm: number;
  private l2Lambda: number;
  private timestep: number = 0;
  private states: Map<string, AdamState> = new Map();

  constructor(config: {
    learningRate: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    gradientClipNorm: number;
    l2Lambda: number;
  }) {
    this.lr = config.learningRate;
    this.beta1 = config.beta1;
    this.beta2 = config.beta2;
    this.epsilon = config.epsilon;
    this.clipNorm = config.gradientClipNorm;
    this.l2Lambda = config.l2Lambda;
  }

  registerParameter(name: string, size: number): void {
    this.states.set(name, new AdamState(size));
  }

  /**
   * Perform optimization step
   * @param parameters - Map of parameter name to {params, grads} buffers
   */
  step(
    parameters: Map<
      string,
      { params: Float64Array; grads: GradientAccumulator }
    >,
  ): void {
    this.timestep++;

    for (const [name, { params, grads }] of parameters) {
      const state = this.states.get(name);
      if (!state) continue;

      // Apply gradient clipping
      grads.clipByNorm(this.clipNorm);

      // Add L2 regularization gradient
      if (this.l2Lambda > 0) {
        LossFunction.l2Gradient(
          grads.gradients,
          0,
          params,
          0,
          this.l2Lambda,
          grads.size,
        );
      }

      // Adam update
      state.update(
        params,
        0,
        grads.gradients,
        0,
        this.lr,
        this.beta1,
        this.beta2,
        this.epsilon,
        this.timestep,
      );

      // Zero gradients
      grads.zero();
    }
  }

  reset(): void {
    this.timestep = 0;
    for (const state of this.states.values()) {
      state.reset();
    }
  }

  getTimestep(): number {
    return this.timestep;
  }

  serialize(): object {
    const statesObj: { [key: string]: any } = {};
    for (const [name, state] of this.states) {
      statesObj[name] = state.serialize();
    }
    return { timestep: this.timestep, states: statesObj };
  }

  deserialize(data: any): void {
    this.timestep = data.timestep || 0;
    for (const [name, stateData] of Object.entries(data.states || {})) {
      const state = this.states.get(name);
      if (state) state.deserialize(stateData as any);
    }
  }
}

// ============================================================================
// PHASE 4: CONVOLUTION LAYERS
// ============================================================================

/**
 * Precomputed index lookup table for causal dilated convolution
 * For each output position t and kernel position k, stores input index
 */
class ConvIndexMap {
  readonly indices: Int32Array;
  readonly kernelSize: number;
  readonly outputLen: number;

  /**
   * Build index map for causal convolution
   * @param kernelSize - Convolution kernel size
   * @param dilation - Dilation factor
   * @param inputLen - Input sequence length
   */
  constructor(kernelSize: number, dilation: number, inputLen: number) {
    this.kernelSize = kernelSize;
    this.outputLen = inputLen; // Causal: same output length as input
    this.indices = new Int32Array(this.outputLen * kernelSize);

    // For each output position and kernel position, compute input index
    // Causal convolution: output[t] depends on input[t-k*dilation] for k in 0..kernelSize-1
    for (let t = 0; t < this.outputLen; t++) {
      for (let k = 0; k < kernelSize; k++) {
        const inputIdx = t - k * dilation;
        // Store -1 for positions before start (zero padding)
        this.indices[t * kernelSize + k] = inputIdx >= 0 ? inputIdx : -1;
      }
    }
  }

  getInputIndex(outPos: number, kernelPos: number): number {
    return this.indices[outPos * this.kernelSize + kernelPos];
  }
}

/**
 * Parameters for causal dilated 1D convolution
 */
class CausalConv1DParams {
  readonly weights: Float64Array; // [outChannels, inChannels, kernelSize]
  readonly bias: Float64Array; // [outChannels]
  readonly outChannels: number;
  readonly inChannels: number;
  readonly kernelSize: number;

  constructor(inChannels: number, outChannels: number, kernelSize: number) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.weights = new Float64Array(outChannels * inChannels * kernelSize);
    this.bias = new Float64Array(outChannels);
  }

  /** Xavier/He initialization */
  initialize(rng: RandomGenerator, scale: number): void {
    const fanIn = this.inChannels * this.kernelSize;
    const std = scale * Math.sqrt(2.0 / fanIn);
    const limit = 2 * std;

    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = rng.truncatedGaussian(std, limit);
    }
    this.bias.fill(0);
  }

  getWeightIndex(outC: number, inC: number, k: number): number {
    return outC * (this.inChannels * this.kernelSize) + inC * this.kernelSize +
      k;
  }
}

/**
 * Causal dilated 1D convolution layer
 */
class CausalConv1D {
  readonly params: CausalConv1DParams;
  readonly weightGrads: GradientAccumulator;
  readonly biasGrads: GradientAccumulator;
  private indexMap: ConvIndexMap | null = null;
  private dilation: number;
  private maxSeqLen: number;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    dilation: number,
    maxSeqLen: number,
  ) {
    this.params = new CausalConv1DParams(inChannels, outChannels, kernelSize);
    this.weightGrads = new GradientAccumulator(
      outChannels * inChannels * kernelSize,
    );
    this.biasGrads = new GradientAccumulator(outChannels);
    this.dilation = dilation;
    this.maxSeqLen = maxSeqLen;
  }

  private ensureIndexMap(seqLen: number): void {
    if (!this.indexMap || this.indexMap.outputLen !== seqLen) {
      this.indexMap = new ConvIndexMap(
        this.params.kernelSize,
        this.dilation,
        seqLen,
      );
    }
  }

  /**
   * Forward pass: output[t,outC] = bias[outC] + sum over inC, k of weight[outC,inC,k] * input[t-k*d, inC]
   * @param output - Output buffer [seqLen, outChannels]
   * @param outOff - Output offset
   * @param input - Input buffer [seqLen, inChannels]
   * @param inOff - Input offset
   * @param seqLen - Sequence length
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    this.ensureIndexMap(seqLen);
    const { outChannels, inChannels, kernelSize, weights, bias } = this.params;

    for (let t = 0; t < seqLen; t++) {
      for (let outC = 0; outC < outChannels; outC++) {
        let sum = bias[outC];

        for (let k = 0; k < kernelSize; k++) {
          const inIdx = this.indexMap!.getInputIndex(t, k);
          if (inIdx >= 0) {
            for (let inC = 0; inC < inChannels; inC++) {
              const wIdx = this.params.getWeightIndex(outC, inC, k);
              sum += weights[wIdx] * input[inOff + inIdx * inChannels + inC];
            }
          }
        }

        output[outOff + t * outChannels + outC] = sum;
      }
    }
  }

  /**
   * Backward pass: compute gradients for weights, bias, and input
   * @param dInput - Gradient w.r.t. input (or null if not needed)
   * @param dInputOff - dInput offset
   * @param dOutput - Gradient w.r.t. output
   * @param dOutOff - dOutput offset
   * @param input - Input from forward pass (for weight gradients)
   * @param inOff - Input offset
   * @param seqLen - Sequence length
   */
  backward(
    dInput: Float64Array | null,
    dInputOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    this.ensureIndexMap(seqLen);
    const { outChannels, inChannels, kernelSize, weights } = this.params;

    // Zero dInput if provided
    if (dInput) {
      TensorOps.fill(dInput, dInputOff, seqLen * inChannels, 0);
    }

    for (let t = 0; t < seqLen; t++) {
      for (let outC = 0; outC < outChannels; outC++) {
        const dOut = dOutput[dOutOff + t * outChannels + outC];

        // Bias gradient
        this.biasGrads.gradients[outC] += dOut;

        for (let k = 0; k < kernelSize; k++) {
          const inIdx = this.indexMap!.getInputIndex(t, k);
          if (inIdx >= 0) {
            for (let inC = 0; inC < inChannels; inC++) {
              const wIdx = this.params.getWeightIndex(outC, inC, k);
              const inVal = input[inOff + inIdx * inChannels + inC];

              // Weight gradient: dL/dW = sum over t of dL/dOut * input
              this.weightGrads.gradients[wIdx] += dOut * inVal;

              // Input gradient: dL/dInput = sum over outC, k of dL/dOut * weight
              if (dInput) {
                dInput[dInputOff + inIdx * inChannels + inC] += dOut *
                  weights[wIdx];
              }
            }
          }
        }
      }
    }
  }

  getParameterCount(): number {
    return this.params.weights.length + this.params.bias.length;
  }
}

/**
 * 1x1 convolution for channel projection (pointwise)
 */
class Conv1x1 {
  readonly weights: Float64Array; // [outChannels, inChannels]
  readonly bias: Float64Array; // [outChannels]
  readonly weightGrads: GradientAccumulator;
  readonly biasGrads: GradientAccumulator;
  readonly outChannels: number;
  readonly inChannels: number;

  constructor(inChannels: number, outChannels: number) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.weights = new Float64Array(outChannels * inChannels);
    this.bias = new Float64Array(outChannels);
    this.weightGrads = new GradientAccumulator(outChannels * inChannels);
    this.biasGrads = new GradientAccumulator(outChannels);
  }

  initialize(rng: RandomGenerator, scale: number): void {
    const std = scale * Math.sqrt(2.0 / this.inChannels);
    const limit = 2 * std;
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = rng.truncatedGaussian(std, limit);
    }
    this.bias.fill(0);
  }

  /** Forward: output[t] = W @ input[t] + bias */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      for (let outC = 0; outC < this.outChannels; outC++) {
        let sum = this.bias[outC];
        for (let inC = 0; inC < this.inChannels; inC++) {
          sum += this.weights[outC * this.inChannels + inC] *
            input[inOff + t * this.inChannels + inC];
        }
        output[outOff + t * this.outChannels + outC] = sum;
      }
    }
  }

  backward(
    dInput: Float64Array | null,
    dInputOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    if (dInput) {
      TensorOps.fill(dInput, dInputOff, seqLen * this.inChannels, 0);
    }

    for (let t = 0; t < seqLen; t++) {
      for (let outC = 0; outC < this.outChannels; outC++) {
        const dOut = dOutput[dOutOff + t * this.outChannels + outC];
        this.biasGrads.gradients[outC] += dOut;

        for (let inC = 0; inC < this.inChannels; inC++) {
          const wIdx = outC * this.inChannels + inC;
          this.weightGrads.gradients[wIdx] += dOut *
            input[inOff + t * this.inChannels + inC];

          if (dInput) {
            dInput[dInputOff + t * this.inChannels + inC] += dOut *
              this.weights[wIdx];
          }
        }
      }
    }
  }

  getParameterCount(): number {
    return this.weights.length + this.bias.length;
  }
}

// ============================================================================
// PHASE 5: TCN BLOCKS
// ============================================================================

/**
 * Dropout mask with preallocated buffer
 */
class DropoutMask {
  private mask: Float64Array;
  private rate: number;

  constructor(maxSize: number, rate: number = 0) {
    this.mask = new Float64Array(maxSize);
    this.rate = rate;
  }

  generate(rng: RandomGenerator, size: number): void {
    if (this.rate === 0) {
      TensorOps.fill(this.mask, 0, size, 1.0);
      return;
    }

    const scale = 1.0 / (1.0 - this.rate);
    for (let i = 0; i < size; i++) {
      this.mask[i] = rng.nextFloat() >= this.rate ? scale : 0;
    }
  }

  applyForward(data: Float64Array, offset: number, size: number): void {
    for (let i = 0; i < size; i++) {
      data[offset + i] *= this.mask[i];
    }
  }

  applyBackward(grad: Float64Array, offset: number, size: number): void {
    for (let i = 0; i < size; i++) {
      grad[offset + i] *= this.mask[i];
    }
  }
}

/**
 * Layer normalization parameters
 */
class LayerNormParams {
  readonly gamma: Float64Array;
  readonly beta: Float64Array;
  readonly gammaGrads: GradientAccumulator;
  readonly betaGrads: GradientAccumulator;
  readonly channels: number;

  constructor(channels: number) {
    this.channels = channels;
    this.gamma = new Float64Array(channels);
    this.beta = new Float64Array(channels);
    this.gamma.fill(1.0);
    this.beta.fill(0.0);
    this.gammaGrads = new GradientAccumulator(channels);
    this.betaGrads = new GradientAccumulator(channels);
  }
}

/**
 * Layer normalization forward/backward ops
 */
class LayerNormOps {
  private static readonly EPSILON = 1e-5;

  /**
   * Forward: for each time step, normalize across channels
   * output = gamma * (input - mean) / sqrt(var + eps) + beta
   */
  static forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    gamma: Float64Array,
    beta: Float64Array,
    stats: Float64Array,
    statsOff: number, // Store mean, variance for backward
    channels: number,
    seqLen: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      // Compute mean
      let mean = 0;
      for (let c = 0; c < channels; c++) {
        mean += input[inOff + t * channels + c];
      }
      mean /= channels;

      // Compute variance
      let variance = 0;
      for (let c = 0; c < channels; c++) {
        const diff = input[inOff + t * channels + c] - mean;
        variance += diff * diff;
      }
      variance /= channels;

      // Store stats for backward
      stats[statsOff + t * 2] = mean;
      stats[statsOff + t * 2 + 1] = variance;

      // Normalize, scale, shift
      const invStd = 1.0 / Math.sqrt(variance + this.EPSILON);
      for (let c = 0; c < channels; c++) {
        const normalized = (input[inOff + t * channels + c] - mean) * invStd;
        output[outOff + t * channels + c] = gamma[c] * normalized + beta[c];
      }
    }
  }

  /**
   * Backward: compute gradients for gamma, beta, and input
   */
  static backward(
    dInput: Float64Array,
    dInputOff: number,
    dGamma: GradientAccumulator,
    dBeta: GradientAccumulator,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
    stats: Float64Array,
    statsOff: number,
    gamma: Float64Array,
    channels: number,
    seqLen: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const mean = stats[statsOff + t * 2];
      const variance = stats[statsOff + t * 2 + 1];
      const invStd = 1.0 / Math.sqrt(variance + this.EPSILON);

      // Compute dGamma, dBeta, and intermediate sums for dInput
      let sumDy = 0;
      let sumDyXhat = 0;

      for (let c = 0; c < channels; c++) {
        const xHat = (input[inOff + t * channels + c] - mean) * invStd;
        const dy = dOutput[dOutOff + t * channels + c];

        dGamma.gradients[c] += dy * xHat;
        dBeta.gradients[c] += dy;

        sumDy += gamma[c] * dy;
        sumDyXhat += gamma[c] * dy * xHat;
      }

      // Compute dInput using the layer norm backward formula
      for (let c = 0; c < channels; c++) {
        const xHat = (input[inOff + t * channels + c] - mean) * invStd;
        const dy = dOutput[dOutOff + t * channels + c];

        dInput[dInputOff + t * channels + c] =
          (gamma[c] * dy - sumDy / channels - xHat * sumDyXhat / channels) *
          invStd;
      }
    }
  }
}

/**
 * Forward context for storing activations needed in backward pass
 */
interface TCNBlockContext {
  preAct1: Float64Array; // Pre-activation after conv1
  preAct1Offset: number;
  postAct1?: Float64Array; // Post-activation (for two-layer blocks)
  postAct1Offset?: number;
  preAct2?: Float64Array; // Pre-activation after conv2
  preAct2Offset?: number;
  input: Float64Array; // Copy of input
  inputOffset: number;
  normStats?: Float64Array; // LayerNorm statistics
  normStatsOffset?: number;
  seqLen: number;
}

/**
 * Single residual TCN block
 * conv1 -> activation -> [conv2 -> activation] -> [norm] -> [dropout] -> residual add
 */
class TCNBlock {
  private conv1: CausalConv1D;
  private conv2: CausalConv1D | null = null;
  private residualProj: Conv1x1 | null = null;
  private layerNorm: LayerNormParams | null = null;
  private dropout: DropoutMask;
  private activation: "relu" | "gelu";
  private useNorm: boolean;
  private useTwoLayer: boolean;
  readonly inChannels: number;
  readonly outChannels: number;

  constructor(config: {
    inChannels: number;
    outChannels: number;
    kernelSize: number;
    dilation: number;
    useTwoLayers: boolean;
    activation: "relu" | "gelu";
    useNorm: boolean;
    dropoutRate: number;
    maxSeqLen: number;
  }) {
    this.inChannels = config.inChannels;
    this.outChannels = config.outChannels;
    this.activation = config.activation;
    this.useNorm = config.useNorm;
    this.useTwoLayer = config.useTwoLayers;

    this.conv1 = new CausalConv1D(
      config.inChannels,
      config.outChannels,
      config.kernelSize,
      config.dilation,
      config.maxSeqLen,
    );

    if (config.useTwoLayers) {
      this.conv2 = new CausalConv1D(
        config.outChannels,
        config.outChannels,
        config.kernelSize,
        config.dilation,
        config.maxSeqLen,
      );
    }

    if (config.inChannels !== config.outChannels) {
      this.residualProj = new Conv1x1(config.inChannels, config.outChannels);
    }

    if (config.useNorm) {
      this.layerNorm = new LayerNormParams(config.outChannels);
    }

    this.dropout = new DropoutMask(
      config.maxSeqLen * config.outChannels,
      config.dropoutRate,
    );
  }

  initialize(rng: RandomGenerator, scale: number): void {
    this.conv1.params.initialize(rng, scale);
    this.conv2?.params.initialize(rng, scale);
    this.residualProj?.initialize(rng, scale);
  }

  /**
   * Forward pass with optional context recording for backward
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
    training: boolean,
    rng: RandomGenerator,
    scratch1: Float64Array,
    scratch2: Float64Array,
    context: TCNBlockContext | null,
  ): void {
    const bufSize = seqLen * this.outChannels;

    // Store input for backward
    if (context) {
      TensorOps.copy(
        context.input,
        context.inputOffset,
        input,
        inOff,
        seqLen * this.inChannels,
      );
      context.seqLen = seqLen;
    }

    // Conv1 -> activation
    this.conv1.forward(scratch1, 0, input, inOff, seqLen);

    if (context) {
      TensorOps.copy(
        context.preAct1,
        context.preAct1Offset,
        scratch1,
        0,
        bufSize,
      );
    }

    this.applyActivation(scratch1, 0, bufSize);

    // Optional conv2 -> activation
    if (this.conv2) {
      if (context && context.postAct1) {
        TensorOps.copy(
          context.postAct1,
          context.postAct1Offset!,
          scratch1,
          0,
          bufSize,
        );
      }

      this.conv2.forward(scratch2, 0, scratch1, 0, seqLen);

      if (context && context.preAct2) {
        TensorOps.copy(
          context.preAct2,
          context.preAct2Offset!,
          scratch2,
          0,
          bufSize,
        );
      }

      this.applyActivation(scratch2, 0, bufSize);

      // Copy result back to scratch1 for next operations
      TensorOps.copy(scratch1, 0, scratch2, 0, bufSize);
    }

    // Optional layer norm
    if (this.layerNorm && context?.normStats) {
      LayerNormOps.forward(
        scratch2,
        0,
        scratch1,
        0,
        this.layerNorm.gamma,
        this.layerNorm.beta,
        context.normStats,
        context.normStatsOffset!,
        this.outChannels,
        seqLen,
      );
      TensorOps.copy(scratch1, 0, scratch2, 0, bufSize);
    }

    // Optional dropout (training only)
    if (training && this.dropout) {
      this.dropout.generate(rng, bufSize);
      this.dropout.applyForward(scratch1, 0, bufSize);
    }

    // Residual connection
    if (this.residualProj) {
      // Project input to match output channels
      this.residualProj.forward(output, outOff, input, inOff, seqLen);
      // Add activation output
      TensorOps.addInPlace(output, outOff, scratch1, 0, bufSize);
    } else {
      // Direct residual (channels match)
      TensorOps.add(output, outOff, scratch1, 0, input, inOff, bufSize);
    }
  }

  /**
   * Backward pass
   */
  backward(
    dInput: Float64Array,
    dInputOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    context: TCNBlockContext,
    scratch1: Float64Array,
    scratch2: Float64Array,
  ): void {
    const seqLen = context.seqLen;
    const bufSize = seqLen * this.outChannels;

    // dOutput comes in for the residual sum
    // Need to split: dActivation and dResidual
    TensorOps.copy(scratch1, 0, dOutput, dOutOff, bufSize);

    // Backward through dropout
    if (this.dropout) {
      this.dropout.applyBackward(scratch1, 0, bufSize);
    }

    // Backward through layer norm
    if (this.layerNorm && context.normStats) {
      // scratch2 will hold d(pre-norm output)
      LayerNormOps.backward(
        scratch2,
        0,
        this.layerNorm.gammaGrads,
        this.layerNorm.betaGrads,
        scratch1,
        0,
        context.preAct2 || context.preAct1,
        context.preAct2Offset ?? context.preAct1Offset,
        context.normStats,
        context.normStatsOffset!,
        this.layerNorm.gamma,
        this.outChannels,
        seqLen,
      );
      TensorOps.copy(scratch1, 0, scratch2, 0, bufSize);
    }

    // Backward through second activation and conv2
    if (this.conv2 && context.preAct2 && context.postAct1) {
      // Backward through activation
      this.applyActivationBackward(
        scratch1,
        0,
        scratch2,
        0,
        context.preAct2,
        context.preAct2Offset!,
        bufSize,
      );

      // Backward through conv2
      this.conv2.backward(
        scratch1,
        0,
        scratch2,
        0,
        context.postAct1,
        context.postAct1Offset!,
        seqLen,
      );

      // scratch1 now contains d(postAct1)
      // Backward through first activation
      this.applyActivationBackward(
        scratch1,
        0,
        scratch2,
        0,
        context.preAct1,
        context.preAct1Offset,
        bufSize,
      );
    } else {
      // No conv2, just backward through first activation
      this.applyActivationBackward(
        scratch1,
        0,
        scratch2,
        0,
        context.preAct1,
        context.preAct1Offset,
        bufSize,
      );
    }

    // Backward through conv1
    // First, need temporary buffer for conv1's dInput
    const conv1DInput = scratch1;
    this.conv1.backward(
      conv1DInput,
      0,
      scratch2,
      0,
      context.input,
      context.inputOffset,
      seqLen,
    );

    // Handle residual connection gradient
    if (this.residualProj) {
      // Backward through residual projection
      this.residualProj.backward(
        scratch2,
        0,
        dOutput,
        dOutOff,
        context.input,
        context.inputOffset,
        seqLen,
      );
      // Add conv1 gradient
      TensorOps.add(
        dInput,
        dInputOff,
        conv1DInput,
        0,
        scratch2,
        0,
        seqLen * this.inChannels,
      );
    } else {
      // Direct residual - just add gradients
      TensorOps.add(
        dInput,
        dInputOff,
        conv1DInput,
        0,
        dOutput,
        dOutOff,
        seqLen * this.inChannels,
      );
    }
  }

  private applyActivation(
    data: Float64Array,
    offset: number,
    len: number,
  ): void {
    if (this.activation === "relu") {
      ActivationOps.reluInPlace(data, offset, len);
    } else {
      ActivationOps.geluInPlace(data, offset, len);
    }
  }

  private applyActivationBackward(
    dIn: Float64Array,
    dInOff: number,
    dOut: Float64Array,
    dOutOff: number,
    preAct: Float64Array,
    preActOff: number,
    len: number,
  ): void {
    if (this.activation === "relu") {
      ActivationOps.reluBackward(
        dIn,
        dInOff,
        dOut,
        dOutOff,
        preAct,
        preActOff,
        len,
      );
    } else {
      ActivationOps.geluBackward(
        dIn,
        dInOff,
        dOut,
        dOutOff,
        preAct,
        preActOff,
        len,
      );
    }
  }

  getParameters(): Map<
    string,
    { params: Float64Array; grads: GradientAccumulator }
  > {
    const params = new Map<
      string,
      { params: Float64Array; grads: GradientAccumulator }
    >();
    params.set("conv1.weights", {
      params: this.conv1.params.weights,
      grads: this.conv1.weightGrads,
    });
    params.set("conv1.bias", {
      params: this.conv1.params.bias,
      grads: this.conv1.biasGrads,
    });

    if (this.conv2) {
      params.set("conv2.weights", {
        params: this.conv2.params.weights,
        grads: this.conv2.weightGrads,
      });
      params.set("conv2.bias", {
        params: this.conv2.params.bias,
        grads: this.conv2.biasGrads,
      });
    }

    if (this.residualProj) {
      params.set("residual.weights", {
        params: this.residualProj.weights,
        grads: this.residualProj.weightGrads,
      });
      params.set("residual.bias", {
        params: this.residualProj.bias,
        grads: this.residualProj.biasGrads,
      });
    }

    if (this.layerNorm) {
      params.set("norm.gamma", {
        params: this.layerNorm.gamma,
        grads: this.layerNorm.gammaGrads,
      });
      params.set("norm.beta", {
        params: this.layerNorm.beta,
        grads: this.layerNorm.betaGrads,
      });
    }

    return params;
  }

  getParameterCount(): number {
    let count = this.conv1.getParameterCount();
    if (this.conv2) count += this.conv2.getParameterCount();
    if (this.residualProj) count += this.residualProj.getParameterCount();
    if (this.layerNorm) count += this.layerNorm.channels * 2;
    return count;
  }
}

/**
 * Full TCN backbone: stack of TCN blocks with computed dilation schedule
 */
class TCNBackbone {
  private blocks: TCNBlock[];
  private dilations: number[];
  readonly receptiveField: number;
  readonly hiddenChannels: number;
  readonly nFeatures: number;

  constructor(config: {
    nFeatures: number;
    hiddenChannels: number;
    nBlocks: number;
    kernelSize: number;
    dilationBase: number;
    activation: "relu" | "gelu";
    useNorm: boolean;
    dropoutRate: number;
    maxSeqLen: number;
    useTwoLayerBlock: boolean;
  }) {
    this.nFeatures = config.nFeatures;
    this.hiddenChannels = config.hiddenChannels;
    this.blocks = [];
    this.dilations = [];

    // Compute dilation schedule: powers of dilationBase
    for (let i = 0; i < config.nBlocks; i++) {
      this.dilations.push(Math.pow(config.dilationBase, i));
    }

    // Compute receptive field: sum of (kernel-1) * dilation for all convolutions
    // With 2-layer blocks, each block has 2 convolutions
    const convsPerBlock = config.useTwoLayerBlock ? 2 : 1;
    let rf = 1;
    for (const dilation of this.dilations) {
      rf += convsPerBlock * (config.kernelSize - 1) * dilation;
    }
    this.receptiveField = rf;

    // Create blocks
    for (let i = 0; i < config.nBlocks; i++) {
      const inC = i === 0 ? config.nFeatures : config.hiddenChannels;
      this.blocks.push(
        new TCNBlock({
          inChannels: inC,
          outChannels: config.hiddenChannels,
          kernelSize: config.kernelSize,
          dilation: this.dilations[i],
          useTwoLayers: config.useTwoLayerBlock,
          activation: config.activation,
          useNorm: config.useNorm,
          dropoutRate: config.dropoutRate,
          maxSeqLen: config.maxSeqLen,
        }),
      );
    }
  }

  initialize(rng: RandomGenerator, scale: number): void {
    for (const block of this.blocks) {
      block.initialize(rng, scale);
    }
  }

  /**
   * Forward pass through all blocks
   * @returns Final hidden state [seqLen, hiddenChannels]
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
    training: boolean,
    rng: RandomGenerator,
    scratch1: Float64Array,
    scratch2: Float64Array,
    scratch3: Float64Array,
    contexts: TCNBlockContext[] | null,
  ): void {
    let currentInput = input;
    let currentInOff = inOff;
    let currentChannels = this.nFeatures;

    for (let i = 0; i < this.blocks.length; i++) {
      const block = this.blocks[i];
      const isLast = i === this.blocks.length - 1;
      const outBuf = isLast ? output : (i % 2 === 0 ? scratch3 : scratch1);
      const outOffset = isLast ? outOff : 0;

      block.forward(
        outBuf,
        outOffset,
        currentInput,
        currentInOff,
        seqLen,
        training,
        rng,
        scratch1,
        scratch2,
        contexts ? contexts[i] : null,
      );

      currentInput = outBuf;
      currentInOff = outOffset;
      currentChannels = this.hiddenChannels;
    }
  }

  /**
   * Backward pass through all blocks in reverse
   */
  backward(
    dInput: Float64Array,
    dInputOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    contexts: TCNBlockContext[],
    scratch1: Float64Array,
    scratch2: Float64Array,
    scratch3: Float64Array,
  ): void {
    let currentDOutput = dOutput;
    let currentDOutOff = dOutOff;

    for (let i = this.blocks.length - 1; i >= 0; i--) {
      const block = this.blocks[i];
      const isFirst = i === 0;
      const dInBuf = isFirst ? dInput : (i % 2 === 0 ? scratch3 : scratch1);
      const dInOffset = isFirst ? dInputOff : 0;

      block.backward(
        dInBuf,
        dInOffset,
        currentDOutput,
        currentDOutOff,
        contexts[i],
        scratch1,
        scratch2,
      );

      currentDOutput = dInBuf;
      currentDOutOff = dInOffset;
    }
  }

  getParameters(): Map<
    string,
    { params: Float64Array; grads: GradientAccumulator }
  > {
    const allParams = new Map<
      string,
      { params: Float64Array; grads: GradientAccumulator }
    >();

    for (let i = 0; i < this.blocks.length; i++) {
      const blockParams = this.blocks[i].getParameters();
      for (const [name, value] of blockParams) {
        allParams.set(`block${i}.${name}`, value);
      }
    }

    return allParams;
  }

  getParameterCount(): number {
    return this.blocks.reduce((sum, b) => sum + b.getParameterCount(), 0);
  }

  get numBlocks(): number {
    return this.blocks.length;
  }
}

// ============================================================================
// PHASE 6: OUTPUT HEAD
// ============================================================================

/**
 * Linear layer (fully connected)
 */
class LinearLayer {
  readonly weights: Float64Array; // [outDim, inDim]
  readonly bias: Float64Array; // [outDim]
  readonly weightGrads: GradientAccumulator;
  readonly biasGrads: GradientAccumulator;
  readonly inDim: number;
  readonly outDim: number;

  constructor(inDim: number, outDim: number) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.weights = new Float64Array(outDim * inDim);
    this.bias = new Float64Array(outDim);
    this.weightGrads = new GradientAccumulator(outDim * inDim);
    this.biasGrads = new GradientAccumulator(outDim);
  }

  initialize(rng: RandomGenerator, scale: number): void {
    const std = scale * Math.sqrt(2.0 / this.inDim);
    const limit = 2 * std;
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = rng.truncatedGaussian(std, limit);
    }
    this.bias.fill(0);
  }

  /** Forward: output = W @ input + bias */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
  ): void {
    for (let i = 0; i < this.outDim; i++) {
      let sum = this.bias[i];
      for (let j = 0; j < this.inDim; j++) {
        sum += this.weights[i * this.inDim + j] * input[inOff + j];
      }
      output[outOff + i] = sum;
    }
  }

  /** Backward: compute gradients and optionally dInput */
  backward(
    dInput: Float64Array | null,
    dInputOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
  ): void {
    // Weight and bias gradients
    for (let i = 0; i < this.outDim; i++) {
      const dOut = dOutput[dOutOff + i];
      this.biasGrads.gradients[i] += dOut;

      for (let j = 0; j < this.inDim; j++) {
        this.weightGrads.gradients[i * this.inDim + j] += dOut *
          input[inOff + j];
      }
    }

    // Input gradient
    if (dInput) {
      TensorOps.fill(dInput, dInputOff, this.inDim, 0);
      for (let i = 0; i < this.outDim; i++) {
        const dOut = dOutput[dOutOff + i];
        for (let j = 0; j < this.inDim; j++) {
          dInput[dInputOff + j] += dOut * this.weights[i * this.inDim + j];
        }
      }
    }
  }

  getParameterCount(): number {
    return this.weights.length + this.bias.length;
  }
}

/**
 * Multi-horizon output head
 * Direct mode: single linear layer mapping to [nTargets * maxFutureSteps]
 * Recursive mode: linear layer mapping to [nTargets], used iteratively
 */
class MultiHorizonHead {
  private linear: LinearLayer;
  readonly nTargets: number;
  readonly maxFutureSteps: number;
  readonly useDirect: boolean;

  constructor(config: {
    hiddenChannels: number;
    nTargets: number;
    maxFutureSteps: number;
    useDirect: boolean;
  }) {
    this.nTargets = config.nTargets;
    this.maxFutureSteps = config.maxFutureSteps;
    this.useDirect = config.useDirect;

    const outDim = config.useDirect
      ? config.nTargets * config.maxFutureSteps
      : config.nTargets;
    this.linear = new LinearLayer(config.hiddenChannels, outDim);
  }

  initialize(rng: RandomGenerator, scale: number): void {
    this.linear.initialize(rng, scale);
  }

  /**
   * Forward pass for direct multi-horizon prediction
   * @param output - Output buffer [maxFutureSteps * nTargets]
   * @param hidden - Hidden state from backbone (last timestep) [hiddenChannels]
   * @param futureSteps - Number of steps to predict
   */
  forward(
    output: Float64Array,
    outOff: number,
    hidden: Float64Array,
    hiddenOff: number,
    futureSteps: number,
  ): void {
    this.linear.forward(output, outOff, hidden, hiddenOff);
  }

  /**
   * Single-step forward for recursive mode
   */
  forwardSingleStep(
    output: Float64Array,
    outOff: number,
    hidden: Float64Array,
    hiddenOff: number,
  ): void {
    this.linear.forward(output, outOff, hidden, hiddenOff);
  }

  backward(
    dHidden: Float64Array,
    dHiddenOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    hidden: Float64Array,
    hiddenOff: number,
  ): void {
    this.linear.backward(
      dHidden,
      dHiddenOff,
      dOutput,
      dOutOff,
      hidden,
      hiddenOff,
    );
  }

  getParameters(): Map<
    string,
    { params: Float64Array; grads: GradientAccumulator }
  > {
    return new Map([
      ["head.weights", {
        params: this.linear.weights,
        grads: this.linear.weightGrads,
      }],
      ["head.bias", { params: this.linear.bias, grads: this.linear.biasGrads }],
    ]);
  }

  getParameterCount(): number {
    return this.linear.getParameterCount();
  }
}

// ============================================================================
// PHASE 7: MODEL ASSEMBLY
// ============================================================================

/**
 * Forward context holding all intermediate values for backward pass
 */
interface ForwardContext {
  blockContexts: TCNBlockContext[];
  lastHidden: Float64Array;
  lastHiddenOffset: number;
  seqLen: number;
}

/**
 * Core TCN model assembling backbone and head
 */
class TCNModel {
  private backbone: TCNBackbone;
  private head: MultiHorizonHead;
  private pool: BufferPool;
  private rng: RandomGenerator;
  readonly nFeatures: number;
  readonly nTargets: number;
  readonly hiddenChannels: number;
  readonly maxSeqLen: number;
  readonly maxFutureSteps: number;

  // Preallocated scratch buffers
  private scratch1: Float64Array;
  private scratch2: Float64Array;
  private scratch3: Float64Array;

  constructor(config: {
    nFeatures: number;
    nTargets: number;
    hiddenChannels: number;
    nBlocks: number;
    kernelSize: number;
    dilationBase: number;
    activation: "relu" | "gelu";
    useNorm: boolean;
    dropoutRate: number;
    maxSeqLen: number;
    maxFutureSteps: number;
    useDirect: boolean;
    weightInitScale: number;
    seed: number;
  }) {
    this.nFeatures = config.nFeatures;
    this.nTargets = config.nTargets;
    this.hiddenChannels = config.hiddenChannels;
    this.maxSeqLen = config.maxSeqLen;
    this.maxFutureSteps = config.maxFutureSteps;

    this.pool = new BufferPool();
    this.rng = new RandomGenerator(config.seed);

    this.backbone = new TCNBackbone({
      nFeatures: config.nFeatures,
      hiddenChannels: config.hiddenChannels,
      nBlocks: config.nBlocks,
      kernelSize: config.kernelSize,
      dilationBase: config.dilationBase,
      activation: config.activation,
      useNorm: config.useNorm,
      dropoutRate: config.dropoutRate,
      maxSeqLen: config.maxSeqLen,
      useTwoLayerBlock: true,
    });

    this.head = new MultiHorizonHead({
      hiddenChannels: config.hiddenChannels,
      nTargets: config.nTargets,
      maxFutureSteps: config.maxFutureSteps,
      useDirect: config.useDirect,
    });

    // Initialize parameters
    this.backbone.initialize(this.rng, config.weightInitScale);
    this.head.initialize(this.rng, config.weightInitScale);

    // Preallocate scratch buffers
    const bufSize = config.maxSeqLen * config.hiddenChannels;
    this.scratch1 = new Float64Array(bufSize);
    this.scratch2 = new Float64Array(bufSize);
    this.scratch3 = new Float64Array(bufSize);
  }

  /**
   * Inference-only forward pass
   * @param output - Output predictions [maxFutureSteps * nTargets]
   * @param input - Input sequence [seqLen * nFeatures]
   * @param seqLen - Actual sequence length
   * @param futureSteps - Number of steps to predict
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
    futureSteps: number,
  ): void {
    // Backbone forward
    const hiddenBuf = this.pool.rent(seqLen * this.hiddenChannels);
    this.backbone.forward(
      hiddenBuf,
      0,
      input,
      inOff,
      seqLen,
      false,
      this.rng,
      this.scratch1,
      this.scratch2,
      this.scratch3,
      null,
    );

    // Extract last timestep hidden state
    const lastHiddenOffset = (seqLen - 1) * this.hiddenChannels;

    // Head forward
    this.head.forward(output, outOff, hiddenBuf, lastHiddenOffset, futureSteps);

    this.pool.return(hiddenBuf);
  }

  /**
   * Training forward pass with context recording
   */
  forwardTrain(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
    futureSteps: number,
    context: ForwardContext,
  ): void {
    // Allocate block contexts
    context.blockContexts = [];
    for (let i = 0; i < this.backbone.numBlocks; i++) {
      const inC = i === 0 ? this.nFeatures : this.hiddenChannels;
      const outC = this.hiddenChannels;
      context.blockContexts.push({
        preAct1: new Float64Array(seqLen * outC),
        preAct1Offset: 0,
        postAct1: new Float64Array(seqLen * outC),
        postAct1Offset: 0,
        preAct2: new Float64Array(seqLen * outC),
        preAct2Offset: 0,
        input: new Float64Array(seqLen * inC),
        inputOffset: 0,
        normStats: new Float64Array(seqLen * 2),
        normStatsOffset: 0,
        seqLen: seqLen,
      });
    }

    // Backbone forward
    const hiddenBuf = this.pool.rent(seqLen * this.hiddenChannels);
    this.backbone.forward(
      hiddenBuf,
      0,
      input,
      inOff,
      seqLen,
      true,
      this.rng,
      this.scratch1,
      this.scratch2,
      this.scratch3,
      context.blockContexts,
    );

    // Store last hidden state for head backward
    const lastHiddenOffset = (seqLen - 1) * this.hiddenChannels;
    context.lastHidden = new Float64Array(this.hiddenChannels);
    context.lastHiddenOffset = 0;
    TensorOps.copy(
      context.lastHidden,
      0,
      hiddenBuf,
      lastHiddenOffset,
      this.hiddenChannels,
    );
    context.seqLen = seqLen;

    // Head forward
    this.head.forward(output, outOff, hiddenBuf, lastHiddenOffset, futureSteps);

    this.pool.return(hiddenBuf);
  }

  /**
   * Backward pass through entire model
   */
  backward(
    dOutput: Float64Array,
    dOutOff: number,
    context: ForwardContext,
  ): void {
    const seqLen = context.seqLen;

    // Backward through head
    const dHidden = new Float64Array(this.hiddenChannels);
    this.head.backward(
      dHidden,
      0,
      dOutput,
      dOutOff,
      context.lastHidden,
      context.lastHiddenOffset,
    );

    // Expand dHidden to full sequence (only last timestep has gradient)
    const dHiddenFull = this.pool.rent(seqLen * this.hiddenChannels);
    dHiddenFull.fill(0);
    TensorOps.copy(
      dHiddenFull,
      (seqLen - 1) * this.hiddenChannels,
      dHidden,
      0,
      this.hiddenChannels,
    );

    // Backward through backbone
    const dInput = this.pool.rent(seqLen * this.nFeatures);
    this.backbone.backward(
      dInput,
      0,
      dHiddenFull,
      0,
      context.blockContexts,
      this.scratch1,
      this.scratch2,
      this.scratch3,
    );

    this.pool.return(dHiddenFull);
    this.pool.return(dInput);
  }

  getAllParameters(): Map<
    string,
    { params: Float64Array; grads: GradientAccumulator }
  > {
    const allParams = new Map<
      string,
      { params: Float64Array; grads: GradientAccumulator }
    >();

    for (const [name, value] of this.backbone.getParameters()) {
      allParams.set(`backbone.${name}`, value);
    }

    for (const [name, value] of this.head.getParameters()) {
      allParams.set(name, value);
    }

    return allParams;
  }

  zeroGradients(): void {
    for (const { grads } of this.getAllParameters().values()) {
      grads.zero();
    }
  }

  getParameterCount(): number {
    return this.backbone.getParameterCount() + this.head.getParameterCount();
  }

  getReceptiveField(): number {
    return this.backbone.receptiveField;
  }

  reinitialize(seed: number, scale: number): void {
    this.rng = new RandomGenerator(seed);
    this.backbone.initialize(this.rng, scale);
    this.head.initialize(this.rng, scale);
  }
}

// ============================================================================
// PHASE 8: TRAINING UTILITIES
// ============================================================================

/**
 * Fixed-size circular buffer for input sequence history
 */
class RingBuffer {
  private data: Float64Array;
  private head: number = 0;
  private count: number = 0;
  private maxLen: number;
  private nFeatures: number;

  constructor(maxLen: number, nFeatures: number) {
    this.maxLen = maxLen;
    this.nFeatures = nFeatures;
    this.data = new Float64Array(maxLen * nFeatures);
  }

  /**
   * Push a new timestep to the buffer
   * @param features - Feature vector for this timestep
   */
  push(features: Float64Array, offset: number): void {
    TensorOps.copy(
      this.data,
      this.head * this.nFeatures,
      features,
      offset,
      this.nFeatures,
    );
    this.head = (this.head + 1) % this.maxLen;
    if (this.count < this.maxLen) this.count++;
  }

  getLength(): number {
    return this.count;
  }

  /**
   * Get last N timesteps as contiguous array
   * @param output - Output buffer [len * nFeatures]
   * @param len - Number of timesteps to retrieve
   */
  getWindow(output: Float64Array, outOff: number, len: number): void {
    const actualLen = Math.min(len, this.count);

    // Calculate start position
    let readPos = (this.head - actualLen + this.maxLen) % this.maxLen;

    // Handle wraparound
    for (let i = 0; i < actualLen; i++) {
      TensorOps.copy(
        output,
        outOff + i * this.nFeatures,
        this.data,
        readPos * this.nFeatures,
        this.nFeatures,
      );
      readPos = (readPos + 1) % this.maxLen;
    }
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
    this.data.fill(0);
  }

  serialize(): object {
    return {
      data: Array.from(this.data),
      head: this.head,
      count: this.count,
    };
  }

  deserialize(obj: any): void {
    for (let i = 0; i < obj.data.length && i < this.data.length; i++) {
      this.data[i] = obj.data[i];
    }
    this.head = obj.head;
    this.count = obj.count;
  }
}

/**
 * Tracks prediction residuals for uncertainty estimation
 */
class ResidualStatsTracker {
  private residuals: Float64Array;
  private head: number = 0;
  private count: number = 0;
  private windowSize: number;
  private nTargets: number;

  constructor(windowSize: number, nTargets: number) {
    this.windowSize = windowSize;
    this.nTargets = nTargets;
    this.residuals = new Float64Array(windowSize * nTargets);
  }

  addResidual(
    predicted: Float64Array,
    predOff: number,
    actual: Float64Array,
    actualOff: number,
  ): void {
    for (let i = 0; i < this.nTargets; i++) {
      this.residuals[this.head * this.nTargets + i] = actual[actualOff + i] -
        predicted[predOff + i];
    }
    this.head = (this.head + 1) % this.windowSize;
    if (this.count < this.windowSize) this.count++;
  }

  /**
   * Get mean and std of residuals per target
   */
  getStats(): { means: Float64Array; stds: Float64Array } {
    const means = new Float64Array(this.nTargets);
    const stds = new Float64Array(this.nTargets);

    if (this.count === 0) {
      stds.fill(1.0); // Default uncertainty
      return { means, stds };
    }

    // Compute mean
    for (let i = 0; i < this.count; i++) {
      for (let t = 0; t < this.nTargets; t++) {
        means[t] += this.residuals[i * this.nTargets + t];
      }
    }
    for (let t = 0; t < this.nTargets; t++) {
      means[t] /= this.count;
    }

    // Compute std
    for (let i = 0; i < this.count; i++) {
      for (let t = 0; t < this.nTargets; t++) {
        const diff = this.residuals[i * this.nTargets + t] - means[t];
        stds[t] += diff * diff;
      }
    }
    for (let t = 0; t < this.nTargets; t++) {
      stds[t] = Math.sqrt(Math.max(stds[t] / this.count, 1e-8));
    }

    return { means, stds };
  }

  getUncertaintyBounds(
    predictions: Float64Array,
    predOff: number,
    len: number,
    lower: Float64Array,
    lowerOff: number,
    upper: Float64Array,
    upperOff: number,
    multiplier: number,
  ): void {
    const { stds } = this.getStats();

    for (let i = 0; i < len; i++) {
      const targetIdx = i % this.nTargets;
      const margin = multiplier * stds[targetIdx];
      lower[lowerOff + i] = predictions[predOff + i] - margin;
      upper[upperOff + i] = predictions[predOff + i] + margin;
    }
  }

  reset(): void {
    this.head = 0;
    this.count = 0;
    this.residuals.fill(0);
  }

  getConfidence(): number {
    if (this.count < 10) return 0.5;
    const { stds } = this.getStats();
    const avgStd = stds.reduce((a, b) => a + b, 0) / this.nTargets;
    // Higher confidence for lower uncertainty
    return Math.max(0.1, Math.min(0.99, 1 / (1 + avgStd)));
  }

  serialize(): object {
    return {
      residuals: Array.from(this.residuals),
      head: this.head,
      count: this.count,
    };
  }

  deserialize(obj: any): void {
    for (
      let i = 0;
      i < obj.residuals.length && i < this.residuals.length;
      i++
    ) {
      this.residuals[i] = obj.residuals[i];
    }
    this.head = obj.head;
    this.count = obj.count;
  }
}

/**
 * ADWIN (Adaptive Windowing) for concept drift detection
 * Uses hierarchical buckets with exponential compression
 */
class ADWINBucket {
  total: number = 0;
  variance: number = 0;
  count: number = 0;

  add(value: number): void {
    const delta = value - (this.count > 0 ? this.total / this.count : 0);
    this.count++;
    this.total += value;
    if (this.count > 1) {
      this.variance += delta * (value - this.total / this.count);
    }
  }

  merge(other: ADWINBucket): void {
    if (other.count === 0) return;
    if (this.count === 0) {
      this.total = other.total;
      this.variance = other.variance;
      this.count = other.count;
      return;
    }

    const n1 = this.count;
    const n2 = other.count;
    const mean1 = this.total / n1;
    const mean2 = other.total / n2;

    this.total += other.total;
    this.count += other.count;

    const delta = mean2 - mean1;
    this.variance += other.variance + delta * delta * n1 * n2 / this.count;
  }

  getMean(): number {
    return this.count > 0 ? this.total / this.count : 0;
  }

  getVariance(): number {
    return this.count > 1 ? this.variance / (this.count - 1) : 0;
  }

  reset(): void {
    this.total = 0;
    this.variance = 0;
    this.count = 0;
  }
}

/**
 * ADWIN drift detector
 */
class ADWINDetector {
  private delta: number;
  private maxBuckets: number;
  private buckets: ADWINBucket[][];
  private lastChangeDetected: boolean = false;
  private windowMean: number = 0;
  private windowCount: number = 0;

  constructor(delta: number = 0.002, maxBuckets: number = 64) {
    this.delta = delta;
    this.maxBuckets = maxBuckets;
    this.buckets = [[]];
  }

  /**
   * Update with new value and check for drift
   * @param value - New observation
   * @returns Whether drift was detected
   */
  update(value: number): boolean {
    // Add to first level bucket
    const newBucket = new ADWINBucket();
    newBucket.add(value);
    this.buckets[0].push(newBucket);

    // Update window stats
    const prevTotal = this.windowMean * this.windowCount;
    this.windowCount++;
    this.windowMean = (prevTotal + value) / this.windowCount;

    // Compress buckets
    this.compress();

    // Check for drift
    this.lastChangeDetected = this.checkDrift();

    return this.lastChangeDetected;
  }

  private compress(): void {
    for (let level = 0; level < this.buckets.length; level++) {
      const levelBuckets = this.buckets[level];

      // Compress pairs at each level
      while (levelBuckets.length >= 2 * (level + 1) + 1) {
        const b1 = levelBuckets.shift()!;
        const b2 = levelBuckets.shift()!;
        b1.merge(b2);

        if (level + 1 >= this.buckets.length) {
          this.buckets.push([]);
        }
        this.buckets[level + 1].push(b1);
      }
    }

    // Enforce max buckets
    let totalBuckets = this.buckets.reduce(
      (sum, level) => sum + level.length,
      0,
    );
    while (totalBuckets > this.maxBuckets && this.buckets.length > 0) {
      // Remove oldest buckets
      for (let level = this.buckets.length - 1; level >= 0; level--) {
        if (this.buckets[level].length > 0) {
          const removed = this.buckets[level].shift()!;
          this.windowCount -= removed.count;
          if (this.windowCount > 0) {
            this.windowMean =
              (this.windowMean * (this.windowCount + removed.count) -
                removed.total) / this.windowCount;
          }
          break;
        }
      }
      totalBuckets--;
    }
  }

  private checkDrift(): boolean {
    if (this.windowCount < 10) return false;

    // Collect all buckets in order
    const allBuckets: ADWINBucket[] = [];
    for (const level of this.buckets) {
      allBuckets.push(...level);
    }

    if (allBuckets.length < 2) return false;

    // Check for significant difference between window halves
    let leftCount = 0, leftSum = 0;
    let rightCount = this.windowCount,
      rightSum = this.windowMean * this.windowCount;

    for (let i = 0; i < allBuckets.length - 1; i++) {
      const bucket = allBuckets[i];
      leftCount += bucket.count;
      leftSum += bucket.total;
      rightCount -= bucket.count;
      rightSum -= bucket.total;

      if (leftCount < 5 || rightCount < 5) continue;

      const leftMean = leftSum / leftCount;
      const rightMean = rightSum / rightCount;
      const diff = Math.abs(leftMean - rightMean);

      // Hoeffding bound approximation
      const m = 1.0 / (1.0 / leftCount + 1.0 / rightCount);
      const epsilon = Math.sqrt(Math.log(2.0 / this.delta) / (2.0 * m));

      if (diff > epsilon) {
        return true;
      }
    }

    return false;
  }

  hasChange(): boolean {
    return this.lastChangeDetected;
  }

  getCurrentMean(): number {
    return this.windowMean;
  }

  reset(): void {
    this.buckets = [[]];
    this.lastChangeDetected = false;
    this.windowMean = 0;
    this.windowCount = 0;
  }

  serialize(): object {
    return {
      windowMean: this.windowMean,
      windowCount: this.windowCount,
      lastChangeDetected: this.lastChangeDetected,
    };
  }

  deserialize(obj: any): void {
    this.windowMean = obj.windowMean || 0;
    this.windowCount = obj.windowCount || 0;
    this.lastChangeDetected = obj.lastChangeDetected || false;
    // Note: bucket structure is not serialized for simplicity
    this.buckets = [[]];
  }
}

/**
 * Computes sample weight based on prediction error z-score
 */
class OutlierDownweighter {
  private threshold: number;
  private minWeight: number;

  constructor(threshold: number = 3.0, minWeight: number = 0.1) {
    this.threshold = threshold;
    this.minWeight = minWeight;
  }

  /**
   * Compute weight for sample based on error magnitude
   * @param error - Prediction error (actual - predicted)
   * @param errorStd - Standard deviation of errors
   * @returns Weight in [minWeight, 1.0]
   */
  computeWeight(error: number, errorStd: number): number {
    if (errorStd < 1e-8) return 1.0;

    const zScore = Math.abs(error) / errorStd;
    if (zScore <= this.threshold) return 1.0;

    return Math.max(this.minWeight, this.threshold / zScore);
  }
}

/**
 * Tracks running metrics for reporting
 */
class MetricsAccumulator {
  private sumLoss: number = 0;
  private sumAbsError: number = 0;
  private count: number = 0;

  update(loss: number, absError: number): void {
    this.sumLoss += loss;
    this.sumAbsError += absError;
    this.count++;
  }

  getMetrics(): { avgLoss: number; mae: number; sampleCount: number } {
    return {
      avgLoss: this.count > 0 ? this.sumLoss / this.count : 0,
      mae: this.count > 0 ? this.sumAbsError / this.count : 0,
      sampleCount: this.count,
    };
  }

  reset(): void {
    this.sumLoss = 0;
    this.sumAbsError = 0;
    this.count = 0;
  }
}

// ============================================================================
// PHASE 9: MAIN API CLASS
// ============================================================================

function applyDefaults(
  config: TCNRegressionConfig,
): Required<TCNRegressionConfig> {
  return {
    maxSequenceLength: config.maxSequenceLength ?? 64,
    maxFutureSteps: config.maxFutureSteps ?? 1,
    hiddenChannels: config.hiddenChannels ?? 32,
    nBlocks: config.nBlocks ?? 4,
    kernelSize: config.kernelSize ?? 3,
    dilationBase: config.dilationBase ?? 2,
    useTwoLayerBlock: config.useTwoLayerBlock ?? true,
    activation: config.activation ?? "relu",
    useLayerNorm: config.useLayerNorm ?? false,
    dropoutRate: config.dropoutRate ?? 0.0,
    learningRate: config.learningRate ?? 0.001,
    beta1: config.beta1 ?? 0.9,
    beta2: config.beta2 ?? 0.999,
    epsilon: config.epsilon ?? 1e-8,
    l2Lambda: config.l2Lambda ?? 0.0001,
    gradientClipNorm: config.gradientClipNorm ?? 1.0,
    normalizationEpsilon: config.normalizationEpsilon ?? 1e-8,
    normalizationWarmup: config.normalizationWarmup ?? 10,
    outlierThreshold: config.outlierThreshold ?? 3.0,
    outlierMinWeight: config.outlierMinWeight ?? 0.1,
    adwinEnabled: config.adwinEnabled ?? true,
    adwinDelta: config.adwinDelta ?? 0.002,
    adwinMaxBuckets: config.adwinMaxBuckets ?? 64,
    useDirectMultiHorizon: config.useDirectMultiHorizon ?? true,
    residualWindowSize: config.residualWindowSize ?? 100,
    uncertaintyMultiplier: config.uncertaintyMultiplier ?? 1.96,
    weightInitScale: config.weightInitScale ?? 0.1,
    seed: config.seed ?? 42,
    verbose: config.verbose ?? false,
  };
}

/**
 * TCNRegression - Main public API class
 *
 * Temporal Convolutional Network for multivariate time series regression
 * with online learning capabilities.
 *
 * @example
 * ```typescript
 * const tcn = new TCNRegression({
 *   maxSequenceLength: 64,
 *   maxFutureSteps: 5,
 *   hiddenChannels: 32,
 *   nBlocks: 4
 * });
 *
 * // Train incrementally
 * for (const sample of data) {
 *   const result = tcn.fitOnline({
 *     xCoordinates: sample.inputs,
 *     yCoordinates: sample.outputs
 *   });
 *   console.log(`Loss: ${result.loss}`);
 * }
 *
 * // Predict
 * const predictions = tcn.predict(5);
 * console.log(predictions.predictions);
 * ```
 */
export class TCNRegression {
  private config: Required<TCNRegressionConfig>;
  private model: TCNModel | null = null;
  private normalizer: WelfordNormalizer | null = null;
  private optimizer: AdamOptimizer | null = null;
  private ringBuffer: RingBuffer | null = null;
  private residualTracker: ResidualStatsTracker | null = null;
  private adwin: ADWINDetector | null = null;
  private outlierWeighter: OutlierDownweighter;
  private metricsAccumulator: MetricsAccumulator;
  private pool: BufferPool;

  private nFeatures: number = 0;
  private nTargets: number = 0;
  private sampleCount: number = 0;
  private initialized: boolean = false;

  // Preallocated buffers
  private normalizedInput: Float64Array | null = null;
  private normalizedOutput: Float64Array | null = null;
  private inputWindow: Float64Array | null = null;
  private predictions: Float64Array | null = null;
  private dLoss: Float64Array | null = null;

  constructor(config: TCNRegressionConfig = {}) {
    this.config = applyDefaults(config);
    this.outlierWeighter = new OutlierDownweighter(
      this.config.outlierThreshold,
      this.config.outlierMinWeight,
    );
    this.metricsAccumulator = new MetricsAccumulator();
    this.pool = new BufferPool();
  }

  /**
   * Initialize model with detected feature dimensions
   */
  private initializeModel(nFeatures: number, nTargets: number): void {
    this.nFeatures = nFeatures;
    this.nTargets = nTargets;

    // Create model
    this.model = new TCNModel({
      nFeatures,
      nTargets,
      hiddenChannels: this.config.hiddenChannels,
      nBlocks: this.config.nBlocks,
      kernelSize: this.config.kernelSize,
      dilationBase: this.config.dilationBase,
      activation: this.config.activation,
      useNorm: this.config.useLayerNorm,
      dropoutRate: this.config.dropoutRate,
      maxSeqLen: this.config.maxSequenceLength,
      maxFutureSteps: this.config.maxFutureSteps,
      useDirect: this.config.useDirectMultiHorizon,
      weightInitScale: this.config.weightInitScale,
      seed: this.config.seed,
    });

    // Create normalizer
    this.normalizer = new WelfordNormalizer(
      nFeatures,
      nTargets,
      this.config.normalizationEpsilon,
      this.config.normalizationWarmup,
    );

    // Create optimizer and register parameters
    this.optimizer = new AdamOptimizer({
      learningRate: this.config.learningRate,
      beta1: this.config.beta1,
      beta2: this.config.beta2,
      epsilon: this.config.epsilon,
      gradientClipNorm: this.config.gradientClipNorm,
      l2Lambda: this.config.l2Lambda,
    });

    for (const [name, { params }] of this.model.getAllParameters()) {
      this.optimizer.registerParameter(name, params.length);
    }

    // Create ring buffer
    this.ringBuffer = new RingBuffer(this.config.maxSequenceLength, nFeatures);

    // Create residual tracker
    this.residualTracker = new ResidualStatsTracker(
      this.config.residualWindowSize,
      nTargets,
    );

    // Create ADWIN detector if enabled
    if (this.config.adwinEnabled) {
      this.adwin = new ADWINDetector(
        this.config.adwinDelta,
        this.config.adwinMaxBuckets,
      );
    }

    // Preallocate buffers
    this.normalizedInput = new Float64Array(nFeatures);
    this.normalizedOutput = new Float64Array(nTargets);
    this.inputWindow = new Float64Array(
      this.config.maxSequenceLength * nFeatures,
    );
    this.predictions = new Float64Array(this.config.maxFutureSteps * nTargets);
    this.dLoss = new Float64Array(this.config.maxFutureSteps * nTargets);

    this.initialized = true;

    if (this.config.verbose) {
      console.log(
        `TCNRegression initialized: ${nFeatures} features -> ${nTargets} targets`,
      );
      console.log(`Parameters: ${this.model.getParameterCount()}`);
      console.log(`Receptive field: ${this.model.getReceptiveField()}`);
    }
  }

  /**
   * Incremental online training step
   *
   * @param input - Training sample with xCoordinates (inputs) and yCoordinates (outputs)
   * @returns Training metrics including loss, sample weight, and drift detection
   *
   * @example
   * ```typescript
   * const result = tcn.fitOnline({
   *   xCoordinates: [[1.0, 2.0, 3.0]],  // Single timestep, 3 features
   *   yCoordinates: [[4.0, 5.0]]        // Single timestep, 2 targets
   * });
   * ```
   */
  fitOnline(
    input: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = input;

    // Validate input
    if (
      !xCoordinates || !yCoordinates || xCoordinates.length === 0 ||
      yCoordinates.length === 0
    ) {
      throw new Error("xCoordinates and yCoordinates must be non-empty arrays");
    }

    const nFeatures = xCoordinates[0].length;
    const nTargets = yCoordinates[0].length;

    // Initialize on first call
    if (!this.initialized) {
      this.initializeModel(nFeatures, nTargets);
    }

    // Validate dimensions match
    if (nFeatures !== this.nFeatures || nTargets !== this.nTargets) {
      throw new Error(
        `Dimension mismatch: expected ${this.nFeatures} features and ${this.nTargets} targets`,
      );
    }

    // Process each timestep
    let totalLoss = 0;
    let totalWeight = 0;
    let driftDetected = false;

    for (let t = 0; t < xCoordinates.length; t++) {
      const x = xCoordinates[t];
      const y = yCoordinates[t];

      // Update normalization statistics
      this.normalizer!.updateInputStats(x);
      this.normalizer!.updateOutputStats(y);

      // Normalize input and add to ring buffer
      this.normalizer!.normalizeInputs(this.normalizedInput!, 0, x);
      this.ringBuffer!.push(this.normalizedInput!, 0);

      // Skip training if not enough history
      const seqLen = this.ringBuffer!.getLength();
      if (
        seqLen <
          Math.min(
            this.model!.getReceptiveField(),
            this.config.maxSequenceLength,
          )
      ) {
        continue;
      }

      // Get input window
      const windowLen = Math.min(seqLen, this.config.maxSequenceLength);
      this.ringBuffer!.getWindow(this.inputWindow!, 0, windowLen);

      // Normalize target
      this.normalizer!.normalizeOutputs(this.normalizedOutput!, 0, y);

      // Forward pass with gradient tape
      const context: ForwardContext = {
        blockContexts: [],
        lastHidden: new Float64Array(0),
        lastHiddenOffset: 0,
        seqLen: 0,
      };

      this.model!.forwardTrain(
        this.predictions!,
        0,
        this.inputWindow!,
        0,
        windowLen,
        1,
        context,
      );

      // Compute loss (MSE for first step only in online learning)
      const loss = LossFunction.mse(
        this.predictions!,
        0,
        this.normalizedOutput!,
        0,
        nTargets,
      );

      // Compute sample weight based on outlier detection
      const { stds } = this.residualTracker!.getStats();
      const avgStd = stds.reduce((a, b) => a + b, 0) / nTargets || 1;
      const error = Math.sqrt(loss);
      const sampleWeight = this.outlierWeighter.computeWeight(error, avgStd);

      // Compute weighted loss gradient
      LossFunction.mseGradientWeighted(
        this.dLoss!,
        0,
        this.predictions!,
        0,
        this.normalizedOutput!,
        0,
        sampleWeight,
        nTargets,
      );

      // Backward pass
      this.model!.backward(this.dLoss!, 0, context);

      // Optimizer step
      this.optimizer!.step(this.model!.getAllParameters());

      // Update residual tracker (denormalize predictions first)
      const denormPred = this.pool.rent(nTargets);
      this.normalizer!.denormalizeOutputs(
        denormPred,
        0,
        this.predictions!,
        0,
        nTargets,
      );
      const actualY = new Float64Array(y);
      this.residualTracker!.addResidual(denormPred, 0, actualY, 0);
      this.pool.return(denormPred);

      // Update ADWIN
      if (this.adwin) {
        if (this.adwin.update(loss)) {
          driftDetected = true;
        }
      }

      // Update metrics
      const absError = Math.sqrt(loss);
      this.metricsAccumulator.update(loss, absError);

      totalLoss += loss * sampleWeight;
      totalWeight += sampleWeight;
      this.sampleCount++;
    }

    const avgLoss = totalWeight > 0 ? totalLoss / totalWeight : 0;
    const avgWeight = xCoordinates.length > 0
      ? totalWeight / xCoordinates.length
      : 1;

    return {
      loss: avgLoss,
      sampleWeight: avgWeight,
      driftDetected,
      metrics: this.metricsAccumulator.getMetrics(),
    };
  }

  /**
   * Generate predictions for future timesteps
   *
   * @param futureSteps - Number of future steps to predict (1 to maxFutureSteps)
   * @returns Predictions with uncertainty bounds
   *
   * @example
   * ```typescript
   * const result = tcn.predict(5);
   * console.log(result.predictions);     // [[pred1], [pred2], ...]
   * console.log(result.uncertaintyLower); // Lower confidence bounds
   * console.log(result.uncertaintyUpper); // Upper confidence bounds
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    if (
      !this.initialized || !this.model || !this.ringBuffer || !this.normalizer
    ) {
      throw new Error("Model not initialized. Call fitOnline first.");
    }

    if (futureSteps < 1 || futureSteps > this.config.maxFutureSteps) {
      throw new Error(
        `futureSteps must be between 1 and ${this.config.maxFutureSteps}`,
      );
    }

    const seqLen = this.ringBuffer.getLength();
    if (seqLen === 0) {
      throw new Error("No input history available. Call fitOnline first.");
    }

    // Get input window
    const windowLen = Math.min(seqLen, this.config.maxSequenceLength);
    this.ringBuffer.getWindow(this.inputWindow!, 0, windowLen);

    // Forward pass
    this.model.forward(
      this.predictions!,
      0,
      this.inputWindow!,
      0,
      windowLen,
      futureSteps,
    );

    // Denormalize predictions and compute uncertainty
    const predictions: number[][] = [];
    const uncertaintyLower: number[][] = [];
    const uncertaintyUpper: number[][] = [];

    const denormPred = this.pool.rent(this.nTargets);
    const lower = this.pool.rent(this.nTargets);
    const upper = this.pool.rent(this.nTargets);

    for (let step = 0; step < futureSteps; step++) {
      const offset = step * this.nTargets;

      // Denormalize predictions
      this.normalizer.denormalizeOutputs(
        denormPred,
        0,
        this.predictions!,
        offset,
        this.nTargets,
      );

      // Compute uncertainty bounds
      this.residualTracker!.getUncertaintyBounds(
        denormPred,
        0,
        this.nTargets,
        lower,
        0,
        upper,
        0,
        this.config.uncertaintyMultiplier,
      );

      predictions.push(Array.from(denormPred.subarray(0, this.nTargets)));
      uncertaintyLower.push(Array.from(lower.subarray(0, this.nTargets)));
      uncertaintyUpper.push(Array.from(upper.subarray(0, this.nTargets)));
    }

    this.pool.return(denormPred);
    this.pool.return(lower);
    this.pool.return(upper);

    const confidence = this.residualTracker!.getConfidence();

    return {
      predictions,
      uncertaintyLower,
      uncertaintyUpper,
      confidence,
    };
  }

  /**
   * Get model architecture summary
   * @returns Detailed model information including parameter counts
   */
  getModelSummary(): ModelSummary {
    const layerParameters: { [key: string]: number } = {};

    if (this.model) {
      for (const [name, { params }] of this.model.getAllParameters()) {
        layerParameters[name] = params.length;
      }
    }

    const totalParameters = this.model?.getParameterCount() || 0;
    const receptiveField = this.model?.getReceptiveField() || 0;

    // Estimate memory usage
    const paramBytes = totalParameters * 8; // Float64
    const momentBytes = totalParameters * 8 * 2; // Adam m and v
    const bufferBytes = this.config.maxSequenceLength *
      this.config.hiddenChannels * 8 * 3;
    const memoryUsageBytes = paramBytes + momentBytes + bufferBytes;

    const architecture = [
      `TCN Regression Model`,
      `  Input features: ${this.nFeatures}`,
      `  Output targets: ${this.nTargets}`,
      `  Hidden channels: ${this.config.hiddenChannels}`,
      `  Blocks: ${this.config.nBlocks}`,
      `  Kernel size: ${this.config.kernelSize}`,
      `  Dilation base: ${this.config.dilationBase}`,
      `  Activation: ${this.config.activation}`,
      `  Layer norm: ${this.config.useLayerNorm}`,
      `  Two-layer blocks: ${this.config.useTwoLayerBlock}`,
      `  Direct multi-horizon: ${this.config.useDirectMultiHorizon}`,
      `  Max sequence length: ${this.config.maxSequenceLength}`,
      `  Max future steps: ${this.config.maxFutureSteps}`,
      `  Receptive field: ${receptiveField}`,
    ].join("\n");

    return {
      architecture,
      totalParameters,
      layerParameters,
      receptiveField,
      memoryUsageBytes,
      config: this.config,
    };
  }

  /**
   * Get all model weights
   * @returns Weight tensors organized by layer
   */
  getWeights(): WeightInfo {
    const weights: WeightInfo = {};

    if (!this.model) return weights;

    for (const [name, { params }] of this.model.getAllParameters()) {
      // Parse shape from parameter name
      const shape = this.inferShape(name, params.length);
      weights[name] = {
        weights: Array.from(params),
        shape,
      };
    }

    return weights;
  }

  private inferShape(name: string, length: number): number[] {
    // Infer shape based on parameter name and length
    if (
      name.includes("bias") || name.includes("beta") || name.includes("gamma")
    ) {
      return [length];
    }
    if (name.includes("conv1.weights") || name.includes("conv2.weights")) {
      // [outChannels, inChannels, kernelSize]
      const k = this.config.kernelSize;
      const h = this.config.hiddenChannels;
      const remaining = length / (h * k);
      return [h, remaining, k];
    }
    if (name.includes("residual.weights") || name.includes("head.weights")) {
      // [outDim, inDim]
      const h = this.config.hiddenChannels;
      if (length % h === 0) {
        return [length / h, h];
      }
    }
    return [length];
  }

  /**
   * Get normalization statistics
   * @returns Current mean/std values for inputs and outputs
   */
  getNormalizationStats(): NormalizationStats {
    if (!this.normalizer) {
      return {
        inputMeans: [],
        inputStds: [],
        outputMeans: [],
        outputStds: [],
        sampleCount: 0,
        warmupComplete: false,
      };
    }
    return this.normalizer.getStats();
  }

  /**
   * Reset model to initial state
   * Re-initializes weights, clears history, resets optimizer
   */
  reset(): void {
    if (this.model) {
      this.model.reinitialize(this.config.seed, this.config.weightInitScale);
      this.model.zeroGradients();
    }

    this.normalizer?.reset();
    this.optimizer?.reset();
    this.ringBuffer?.clear();
    this.residualTracker?.reset();
    this.adwin?.reset();
    this.metricsAccumulator.reset();
    this.sampleCount = 0;

    if (this.config.verbose) {
      console.log("TCNRegression reset");
    }
  }

  /**
   * Serialize model state to JSON string
   * @returns JSON string containing all model state
   */
  save(): string {
    if (!this.initialized) {
      throw new Error("Model not initialized. Nothing to save.");
    }

    const state: any = {
      version: "1.0.0",
      config: this.config,
      nFeatures: this.nFeatures,
      nTargets: this.nTargets,
      sampleCount: this.sampleCount,

      // Model parameters
      parameters: {} as { [key: string]: number[] },

      // Optimizer state
      optimizer: this.optimizer?.serialize(),

      // Normalizer state
      normalizer: this.normalizer?.serialize(),

      // Ring buffer
      ringBuffer: this.ringBuffer?.serialize(),

      // Residual tracker
      residualTracker: this.residualTracker?.serialize(),

      // ADWIN
      adwin: this.adwin?.serialize(),
    };

    // Save parameters
    if (this.model) {
      for (const [name, { params }] of this.model.getAllParameters()) {
        state.parameters[name] = Array.from(params);
      }
    }

    return JSON.stringify(state);
  }

  /**
   * Load model state from JSON string
   * @param jsonStr - JSON string from save()
   */
  load(jsonStr: string): void {
    const state = JSON.parse(jsonStr);

    if (!state.version) {
      throw new Error("Invalid save format: missing version");
    }

    // Restore config and initialize
    this.config = state.config;
    this.initializeModel(state.nFeatures, state.nTargets);
    this.sampleCount = state.sampleCount;

    // Restore parameters
    if (this.model && state.parameters) {
      for (const [name, { params }] of this.model.getAllParameters()) {
        const savedParams = state.parameters[name];
        if (savedParams) {
          for (let i = 0; i < params.length && i < savedParams.length; i++) {
            params[i] = savedParams[i];
          }
        }
      }
    }

    // Restore optimizer
    if (this.optimizer && state.optimizer) {
      this.optimizer.deserialize(state.optimizer);
    }

    // Restore normalizer
    if (this.normalizer && state.normalizer) {
      this.normalizer.deserialize(state.normalizer);
    }

    // Restore ring buffer
    if (this.ringBuffer && state.ringBuffer) {
      this.ringBuffer.deserialize(state.ringBuffer);
    }

    // Restore residual tracker
    if (this.residualTracker && state.residualTracker) {
      this.residualTracker.deserialize(state.residualTracker);
    }

    // Restore ADWIN
    if (this.adwin && state.adwin) {
      this.adwin.deserialize(state.adwin);
    }

    if (this.config.verbose) {
      console.log("TCNRegression loaded from save");
    }
  }
}

// Export default
export default TCNRegression;
