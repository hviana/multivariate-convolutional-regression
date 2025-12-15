// TCNRegression.ts - Complete Implementation

// ==================== TYPE DEFINITIONS ====================

/**
 * Configuration for TCNRegression model
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

/** Result returned by fitOnline */
export interface FitResult {
  loss: number;
  sampleWeight: number;
  driftDetected: boolean;
  metrics: { avgLoss: number; mae: number; sampleCount: number };
}

/** Result returned by predict */
export interface PredictionResult {
  predictions: number[][];
  uncertaintyLower: number[][];
  uncertaintyUpper: number[][];
  confidence: number;
}

/** Model summary information */
export interface ModelSummary {
  architecture: string;
  layerParams: { [name: string]: number };
  totalParams: number;
  receptiveField: number;
  memoryBytes: number;
  nFeatures: number;
  nTargets: number;
  sampleCount: number;
}

/** Weight information for inspection */
export interface WeightInfo {
  parameters: {
    [layerName: string]: {
      weights?: { data: number[]; shape: number[] };
      bias?: { data: number[]; shape: number[] };
    };
  };
}

/** Normalization statistics */
export interface NormalizationStats {
  inputMeans: number[];
  inputStds: number[];
  outputMeans: number[];
  outputStds: number[];
  sampleCount: number;
  isWarmedUp: boolean;
}

// ==================== INTERNAL CONFIGURATION ====================

interface ResolvedConfig {
  maxSequenceLength: number;
  maxFutureSteps: number;
  hiddenChannels: number;
  nBlocks: number;
  kernelSize: number;
  dilationBase: number;
  useTwoLayerBlock: boolean;
  activation: "relu" | "gelu";
  useLayerNorm: boolean;
  dropoutRate: number;
  learningRate: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  l2Lambda: number;
  gradientClipNorm: number;
  normalizationEpsilon: number;
  normalizationWarmup: number;
  outlierThreshold: number;
  outlierMinWeight: number;
  adwinEnabled: boolean;
  adwinDelta: number;
  adwinMaxBuckets: number;
  useDirectMultiHorizon: boolean;
  residualWindowSize: number;
  uncertaintyMultiplier: number;
  weightInitScale: number;
  seed: number;
  verbose: boolean;
  nFeatures: number;
  nTargets: number;
}

// ==================== PHASE 1: MEMORY INFRASTRUCTURE ====================

/**
 * Immutable descriptor holding dimensions and precomputed strides for row-major layout.
 * Validates shape on construction and provides indexing helpers.
 */
class TensorShape {
  readonly dims: readonly number[];
  readonly strides: readonly number[];
  readonly numel: number;

  constructor(dims: number[]) {
    for (let i = 0; i < dims.length; i++) {
      if (!Number.isInteger(dims[i]) || dims[i] <= 0) {
        throw new Error(`Invalid dimension at index ${i}: ${dims[i]}`);
      }
    }
    this.dims = Object.freeze([...dims]);
    const strides = new Array<number>(dims.length);
    let stride = 1;
    for (let i = dims.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= dims[i];
    }
    this.strides = Object.freeze(strides);
    this.numel = stride;
  }

  /** Compute flat offset from multi-index */
  index(...indices: number[]): number {
    let offset = 0;
    for (let i = 0; i < indices.length; i++) {
      offset += indices[i] * this.strides[i];
    }
    return offset;
  }

  dim(i: number): number {
    return this.dims[i];
  }

  get ndim(): number {
    return this.dims.length;
  }

  equals(other: TensorShape): boolean {
    if (this.ndim !== other.ndim) return false;
    for (let i = 0; i < this.ndim; i++) {
      if (this.dims[i] !== other.dims[i]) return false;
    }
    return true;
  }
}

/**
 * Zero-copy view into a Float64Array slab. Holds reference to underlying data,
 * offset, shape, and strides. Provides get/set by multi-index, slice views,
 * and reshape without allocation. Poolable shell object.
 */
class TensorView {
  data: Float64Array;
  offset: number;
  shape: TensorShape;
  private _strides: readonly number[];

  constructor(
    data: Float64Array,
    offset: number,
    shape: TensorShape,
    strides?: readonly number[],
  ) {
    this.data = data;
    this.offset = offset;
    this.shape = shape;
    this._strides = strides ?? shape.strides;
    if (offset + shape.numel > data.length) {
      throw new Error(
        `TensorView out of bounds: offset=${offset}, numel=${shape.numel}, data.length=${data.length}`,
      );
    }
  }

  get(...indices: number[]): number {
    let idx = this.offset;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this._strides[i];
    }
    return this.data[idx];
  }

  set(value: number, ...indices: number[]): void {
    let idx = this.offset;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this._strides[i];
    }
    this.data[idx] = value;
  }

  getFlat(i: number): number {
    return this.data[this.offset + i];
  }

  setFlat(i: number, value: number): void {
    this.data[this.offset + i] = value;
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

  copyFromArray(arr: number[]): void {
    const n = Math.min(arr.length, this.shape.numel);
    for (let i = 0; i < n; i++) {
      this.data[this.offset + i] = arr[i];
    }
  }

  toArray(): number[] {
    const arr = new Array<number>(this.shape.numel);
    for (let i = 0; i < this.shape.numel; i++) {
      arr[i] = this.data[this.offset + i];
    }
    return arr;
  }

  /** Reset view to point to different location (for pooling) */
  reset(data: Float64Array, offset: number, shape: TensorShape): void {
    this.data = data;
    this.offset = offset;
    this.shape = shape;
    this._strides = shape.strides;
  }
}

/**
 * Manages reusable scratch buffers organized by size class (powers of two).
 * Rent returns existing buffer or creates one if pool empty.
 * Return recycles buffer. Tracks high-water mark for diagnostics.
 */
class BufferPool {
  private pools: Map<number, Float64Array[]> = new Map();
  private highWaterMark: Map<number, number> = new Map();
  private static readonly SIZE_CLASSES = [
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
    for (const size of BufferPool.SIZE_CLASSES) {
      if (size >= minSize) return size;
    }
    // Return next power of two for larger sizes
    let size = 65536;
    while (size < minSize) size *= 2;
    return size;
  }

  /** Rent a buffer of at least minSize elements */
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
    const current = (this.highWaterMark.get(sizeClass) ?? 0) + 1;
    this.highWaterMark.set(sizeClass, current);

    return new Float64Array(sizeClass);
  }

  /** Return a buffer to the pool */
  return(buffer: Float64Array): void {
    const sizeClass = buffer.length;
    let pool = this.pools.get(sizeClass);
    if (!pool) {
      pool = [];
      this.pools.set(sizeClass, pool);
    }
    pool.push(buffer);
  }

  /** Get pool statistics */
  getStats(): { sizeClass: number; pooled: number; highWater: number }[] {
    const stats: { sizeClass: number; pooled: number; highWater: number }[] =
      [];
    for (const [size, pool] of this.pools) {
      stats.push({
        sizeClass: size,
        pooled: pool.length,
        highWater: this.highWaterMark.get(size) ?? 0,
      });
    }
    return stats;
  }
}

/**
 * Preallocated contiguous Float64Array slab sized at initialization.
 * Allocates views via bump pointer. Supports mark/release for scoped temporaries.
 * Never grows after init.
 */
class TensorArena {
  private data: Float64Array;
  private offset: number = 0;
  private marks: number[] = [];

  constructor(totalSize: number) {
    this.data = new Float64Array(totalSize);
  }

  /** Allocate a view with given shape */
  alloc(shape: TensorShape): TensorView {
    if (this.offset + shape.numel > this.data.length) {
      throw new Error(
        `TensorArena overflow: needed ${shape.numel}, have ${
          this.data.length - this.offset
        }`,
      );
    }
    const view = new TensorView(this.data, this.offset, shape);
    this.offset += shape.numel;
    return view;
  }

  /** Allocate raw buffer */
  allocRaw(size: number): Float64Array {
    if (this.offset + size > this.data.length) {
      throw new Error(
        `TensorArena overflow: needed ${size}, have ${
          this.data.length - this.offset
        }`,
      );
    }
    const start = this.offset;
    this.offset += size;
    return this.data.subarray(start, this.offset);
  }

  /** Mark current position for later release */
  mark(): number {
    this.marks.push(this.offset);
    return this.offset;
  }

  /** Release back to marked position */
  release(marker?: number): void {
    if (marker !== undefined) {
      this.offset = marker;
      while (
        this.marks.length > 0 && this.marks[this.marks.length - 1] > marker
      ) {
        this.marks.pop();
      }
    } else if (this.marks.length > 0) {
      this.offset = this.marks.pop()!;
    }
  }

  /** Reset to beginning */
  reset(): void {
    this.offset = 0;
    this.marks = [];
  }

  /** Get current usage */
  getUsage(): { used: number; total: number } {
    return { used: this.offset, total: this.data.length };
  }
}

/**
 * Static utility class with low-level tensor operations.
 * All operate on raw arrays with explicit offsets. No allocations.
 */
class TensorOps {
  static fill(
    data: Float64Array,
    offset: number,
    length: number,
    value: number,
  ): void {
    const end = offset + length;
    for (let i = offset; i < end; i++) {
      data[i] = value;
    }
  }

  static copy(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
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
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      dst[dstOff + i] = a[aOff + i] + b[bOff + i];
    }
  }

  static addInPlace(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      dst[dstOff + i] += src[srcOff + i];
    }
  }

  static scale(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    s: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      dst[dstOff + i] = src[srcOff + i] * s;
    }
  }

  static scaleInPlace(
    data: Float64Array,
    offset: number,
    s: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      data[offset + i] *= s;
    }
  }

  static axpy(
    dst: Float64Array,
    dstOff: number,
    a: number,
    x: Float64Array,
    xOff: number,
    y: Float64Array,
    yOff: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      dst[dstOff + i] = a * x[xOff + i] + y[yOff + i];
    }
  }

  static dot(
    a: Float64Array,
    aOff: number,
    b: Float64Array,
    bOff: number,
    length: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < length; i++) {
      sum += a[aOff + i] * b[bOff + i];
    }
    return sum;
  }

  static sum(data: Float64Array, offset: number, length: number): number {
    let sum = 0;
    for (let i = 0; i < length; i++) {
      sum += data[offset + i];
    }
    return sum;
  }

  static max(data: Float64Array, offset: number, length: number): number {
    let max = data[offset];
    for (let i = 1; i < length; i++) {
      if (data[offset + i] > max) max = data[offset + i];
    }
    return max;
  }

  static l2Norm(data: Float64Array, offset: number, length: number): number {
    let sum = 0;
    for (let i = 0; i < length; i++) {
      const v = data[offset + i];
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

  /** Matrix multiply: C[M,N] = A[M,K] @ B[K,N] */
  static matmul(
    C: Float64Array,
    cOff: number,
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
        C[cOff + i * N + j] = sum;
      }
    }
  }

  /** Matrix-vector multiply: y[M] = A[M,K] @ x[K] */
  static matvec(
    y: Float64Array,
    yOff: number,
    A: Float64Array,
    aOff: number,
    x: Float64Array,
    xOff: number,
    M: number,
    K: number,
  ): void {
    for (let i = 0; i < M; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += A[aOff + i * K + k] * x[xOff + k];
      }
      y[yOff + i] = sum;
    }
  }

  /** Transposed matrix-vector multiply: y[K] = A[M,K]^T @ x[M] */
  static matvecT(
    y: Float64Array,
    yOff: number,
    A: Float64Array,
    aOff: number,
    x: Float64Array,
    xOff: number,
    M: number,
    K: number,
  ): void {
    TensorOps.fill(y, yOff, K, 0);
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        y[yOff + k] += A[aOff + i * K + k] * x[xOff + i];
      }
    }
  }

  /** Outer product: A[M,N] += x[M] ⊗ y[N] */
  static outerAdd(
    A: Float64Array,
    aOff: number,
    x: Float64Array,
    xOff: number,
    y: Float64Array,
    yOff: number,
    M: number,
    N: number,
  ): void {
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        A[aOff + i * N + j] += x[xOff + i] * y[yOff + j];
      }
    }
  }
}

// ==================== PHASE 2: NUMERICAL UTILITIES ====================

/**
 * Deterministic PRNG using xorshift128+ algorithm.
 * Seeded at construction. Provides uniform, gaussian, and truncatedGaussian methods.
 */
class RandomGenerator {
  private s0: number;
  private s1: number;
  private hasSpare: boolean = false;
  private spare: number = 0;

  constructor(seed: number) {
    // Initialize state from seed using splitmix64
    let s = seed >>> 0;
    s = ((s >>> 16) ^ s) * 0x45d9f3b >>> 0;
    s = ((s >>> 16) ^ s) * 0x45d9f3b >>> 0;
    s = ((s >>> 16) ^ s) >>> 0;
    this.s0 = s === 0 ? 1 : s;

    s = (seed + 0x9e3779b9) >>> 0;
    s = ((s >>> 16) ^ s) * 0x45d9f3b >>> 0;
    s = ((s >>> 16) ^ s) * 0x45d9f3b >>> 0;
    s = ((s >>> 16) ^ s) >>> 0;
    this.s1 = s === 0 ? 1 : s;
  }

  /** Returns uniform random in [0, 1) */
  nextFloat(): number {
    let s1 = this.s0;
    const s0 = this.s1;
    this.s0 = s0;
    s1 ^= (s1 << 23) >>> 0;
    s1 ^= s1 >>> 17;
    s1 ^= s0;
    s1 ^= s0 >>> 26;
    this.s1 = s1;
    return ((this.s0 + this.s1) >>> 0) / 4294967296;
  }

  /** Returns standard normal using Box-Muller */
  nextGaussian(): number {
    if (this.hasSpare) {
      this.hasSpare = false;
      return this.spare;
    }
    let u: number, v: number, s: number;
    do {
      u = this.nextFloat() * 2 - 1;
      v = this.nextFloat() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt(-2 * Math.log(s) / s);
    this.spare = v * mul;
    this.hasSpare = true;
    return u * mul;
  }

  /** Returns gaussian clipped to [-limit, limit] * std */
  truncatedGaussian(std: number, limit: number = 2): number {
    let val: number;
    do {
      val = this.nextGaussian();
    } while (Math.abs(val) > limit);
    return val * std;
  }
}

/**
 * Static class for activation functions and their derivatives.
 * In-place variants provided. Uses precomputed constants for GELU approximation.
 */
class ActivationOps {
  private static readonly SQRT_2_PI = Math.sqrt(2 / Math.PI);
  private static readonly GELU_COEF = 0.044715;

  static relu(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOff + i] = src[srcOff + i] > 0 ? src[srcOff + i] : 0;
    }
  }

  static reluInPlace(data: Float64Array, offset: number, len: number): void {
    for (let i = 0; i < len; i++) {
      if (data[offset + i] < 0) data[offset + i] = 0;
    }
  }

  static reluBackward(
    dIn: Float64Array,
    dInOff: number,
    dOut: Float64Array,
    dOutOff: number,
    preAct: Float64Array,
    preActOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dIn[dInOff + i] = preAct[preActOff + i] > 0 ? dOut[dOutOff + i] : 0;
    }
  }

  /** GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) */
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
      const inner = ActivationOps.SQRT_2_PI *
        (x + ActivationOps.GELU_COEF * x3);
      dst[dstOff + i] = 0.5 * x * (1 + Math.tanh(inner));
    }
  }

  static geluInPlace(data: Float64Array, offset: number, len: number): void {
    for (let i = 0; i < len; i++) {
      const x = data[offset + i];
      const x3 = x * x * x;
      const inner = ActivationOps.SQRT_2_PI *
        (x + ActivationOps.GELU_COEF * x3);
      data[offset + i] = 0.5 * x * (1 + Math.tanh(inner));
    }
  }

  /** GELU backward (derivative of approximation) */
  static geluBackward(
    dIn: Float64Array,
    dInOff: number,
    dOut: Float64Array,
    dOutOff: number,
    preAct: Float64Array,
    preActOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      const x = preAct[preActOff + i];
      const x2 = x * x;
      const x3 = x2 * x;
      const inner = ActivationOps.SQRT_2_PI *
        (x + ActivationOps.GELU_COEF * x3);
      const tanhInner = Math.tanh(inner);
      const sech2 = 1 - tanhInner * tanhInner;
      const dInner = ActivationOps.SQRT_2_PI *
        (1 + 3 * ActivationOps.GELU_COEF * x2);
      const dGelu = 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * dInner;
      dIn[dInOff + i] = dOut[dOutOff + i] * dGelu;
    }
  }
}

/**
 * Tracks running mean and variance for a single scalar using Welford online algorithm.
 * Numerically stable for streaming data.
 *
 * Formula:
 *   count++
 *   delta = value - mean
 *   mean += delta / count
 *   delta2 = value - mean
 *   m2 += delta * delta2
 */
class WelfordAccumulator {
  private count: number = 0;
  private mean: number = 0;
  private m2: number = 0;
  private epsilon: number;

  constructor(epsilon: number = 1e-8) {
    this.epsilon = epsilon;
  }

  /** Update with new value */
  update(value: number): void {
    if (!Number.isFinite(value)) return;
    this.count++;
    const delta = value - this.mean;
    this.mean += delta / this.count;
    const delta2 = value - this.mean;
    this.m2 += delta * delta2;
  }

  getMean(): number {
    return this.mean;
  }

  getVariance(): number {
    return this.count > 1 ? this.m2 / (this.count - 1) : 0;
  }

  getStd(): number {
    return Math.sqrt(Math.max(this.getVariance(), this.epsilon));
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
 * Manages array of WelfordAccumulator instances for multi-feature normalization.
 * Applies variance floor epsilon.
 */
class WelfordNormalizer {
  private inputAccumulators: WelfordAccumulator[];
  private outputAccumulators: WelfordAccumulator[];
  private epsilon: number;
  private warmupSamples: number;
  private nFeatures: number;
  private nTargets: number;

  constructor(
    nFeatures: number,
    nTargets: number,
    epsilon: number = 1e-8,
    warmupSamples: number = 10,
  ) {
    this.nFeatures = nFeatures;
    this.nTargets = nTargets;
    this.epsilon = epsilon;
    this.warmupSamples = warmupSamples;
    this.inputAccumulators = [];
    this.outputAccumulators = [];
    for (let i = 0; i < nFeatures; i++) {
      this.inputAccumulators.push(new WelfordAccumulator(epsilon));
    }
    for (let i = 0; i < nTargets; i++) {
      this.outputAccumulators.push(new WelfordAccumulator(epsilon));
    }
  }

  /** Update input statistics */
  updateInputStats(features: number[]): void {
    for (let i = 0; i < this.nFeatures && i < features.length; i++) {
      this.inputAccumulators[i].update(features[i]);
    }
  }

  /** Update output statistics */
  updateOutputStats(targets: number[]): void {
    for (let i = 0; i < this.nTargets && i < targets.length; i++) {
      this.outputAccumulators[i].update(targets[i]);
    }
  }

  /** Normalize inputs: z = (x - mean) / std */
  normalizeInputs(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
  ): void {
    const count = this.inputAccumulators[0].getCount();
    if (count < this.warmupSamples) {
      TensorOps.copy(dst, dstOff, src, srcOff, this.nFeatures);
      return;
    }
    for (let i = 0; i < this.nFeatures; i++) {
      const mean = this.inputAccumulators[i].getMean();
      const std = this.inputAccumulators[i].getStd();
      dst[dstOff + i] = (src[srcOff + i] - mean) / std;
    }
  }

  /** Normalize inputs from number array */
  normalizeInputsFromArray(
    dst: Float64Array,
    dstOff: number,
    src: number[],
  ): void {
    const count = this.inputAccumulators[0].getCount();
    if (count < this.warmupSamples) {
      for (let i = 0; i < this.nFeatures; i++) {
        dst[dstOff + i] = src[i];
      }
      return;
    }
    for (let i = 0; i < this.nFeatures; i++) {
      const mean = this.inputAccumulators[i].getMean();
      const std = this.inputAccumulators[i].getStd();
      dst[dstOff + i] = (src[i] - mean) / std;
    }
  }

  /** Normalize outputs */
  normalizeOutputs(dst: Float64Array, dstOff: number, src: number[]): void {
    const count = this.outputAccumulators[0].getCount();
    if (count < this.warmupSamples) {
      for (let i = 0; i < this.nTargets; i++) {
        dst[dstOff + i] = src[i];
      }
      return;
    }
    for (let i = 0; i < this.nTargets; i++) {
      const mean = this.outputAccumulators[i].getMean();
      const std = this.outputAccumulators[i].getStd();
      dst[dstOff + i] = (src[i] - mean) / std;
    }
  }

  /** Denormalize outputs: x = z * std + mean */
  denormalizeOutputs(
    dst: Float64Array,
    dstOff: number,
    src: Float64Array,
    srcOff: number,
  ): void {
    const count = this.outputAccumulators[0].getCount();
    if (count < this.warmupSamples) {
      TensorOps.copy(dst, dstOff, src, srcOff, this.nTargets);
      return;
    }
    for (let i = 0; i < this.nTargets; i++) {
      const mean = this.outputAccumulators[i].getMean();
      const std = this.outputAccumulators[i].getStd();
      dst[dstOff + i] = src[srcOff + i] * std + mean;
    }
  }

  /** Denormalize to number array */
  denormalizeOutputsToArray(src: Float64Array, srcOff: number): number[] {
    const result = new Array<number>(this.nTargets);
    const count = this.outputAccumulators[0].getCount();
    if (count < this.warmupSamples) {
      for (let i = 0; i < this.nTargets; i++) {
        result[i] = src[srcOff + i];
      }
      return result;
    }
    for (let i = 0; i < this.nTargets; i++) {
      const mean = this.outputAccumulators[i].getMean();
      const std = this.outputAccumulators[i].getStd();
      result[i] = src[srcOff + i] * std + mean;
    }
    return result;
  }

  getStats(): NormalizationStats {
    const inputMeans = new Array<number>(this.nFeatures);
    const inputStds = new Array<number>(this.nFeatures);
    const outputMeans = new Array<number>(this.nTargets);
    const outputStds = new Array<number>(this.nTargets);
    for (let i = 0; i < this.nFeatures; i++) {
      inputMeans[i] = this.inputAccumulators[i].getMean();
      inputStds[i] = this.inputAccumulators[i].getStd();
    }
    for (let i = 0; i < this.nTargets; i++) {
      outputMeans[i] = this.outputAccumulators[i].getMean();
      outputStds[i] = this.outputAccumulators[i].getStd();
    }
    const count = this.inputAccumulators[0].getCount();
    return {
      inputMeans,
      inputStds,
      outputMeans,
      outputStds,
      sampleCount: count,
      isWarmedUp: count >= this.warmupSamples,
    };
  }

  isWarmedUp(): boolean {
    return this.inputAccumulators[0].getCount() >= this.warmupSamples;
  }

  getSampleCount(): number {
    return this.inputAccumulators[0].getCount();
  }

  getOutputStd(idx: number): number {
    return this.outputAccumulators[idx].getStd();
  }

  reset(): void {
    for (const acc of this.inputAccumulators) acc.reset();
    for (const acc of this.outputAccumulators) acc.reset();
  }

  serialize(): object {
    return {
      inputs: this.inputAccumulators.map((a) => a.serialize()),
      outputs: this.outputAccumulators.map((a) => a.serialize()),
    };
  }

  deserialize(data: { inputs: any[]; outputs: any[] }): void {
    for (let i = 0; i < this.nFeatures; i++) {
      this.inputAccumulators[i].deserialize(data.inputs[i]);
    }
    for (let i = 0; i < this.nTargets; i++) {
      this.outputAccumulators[i].deserialize(data.outputs[i]);
    }
  }
}

/**
 * Static methods for loss computation and gradients.
 *
 * MSE = (1/n) * Σ(pred - target)²
 * ∂MSE/∂pred = (2/n) * (pred - target)
 */
class LossFunction {
  static mse(
    predictions: Float64Array,
    predOff: number,
    targets: Float64Array,
    targetOff: number,
    len: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const diff = predictions[predOff + i] - targets[targetOff + i];
      sum += diff * diff;
    }
    return sum / len;
  }

  static mseWeighted(
    predictions: Float64Array,
    predOff: number,
    targets: Float64Array,
    targetOff: number,
    weight: number,
    len: number,
  ): number {
    return weight *
      LossFunction.mse(predictions, predOff, targets, targetOff, len);
  }

  static mseGradient(
    dLoss: Float64Array,
    dLossOff: number,
    predictions: Float64Array,
    predOff: number,
    targets: Float64Array,
    targetOff: number,
    len: number,
  ): void {
    const scale = 2 / len;
    for (let i = 0; i < len; i++) {
      dLoss[dLossOff + i] = scale *
        (predictions[predOff + i] - targets[targetOff + i]);
    }
  }

  static mseGradientWeighted(
    dLoss: Float64Array,
    dLossOff: number,
    predictions: Float64Array,
    predOff: number,
    targets: Float64Array,
    targetOff: number,
    weight: number,
    len: number,
  ): void {
    const scale = 2 * weight / len;
    for (let i = 0; i < len; i++) {
      dLoss[dLossOff + i] = scale *
        (predictions[predOff + i] - targets[targetOff + i]);
    }
  }
}

// ==================== PHASE 3: OPTIMIZER ====================

/**
 * Holds gradient buffer matching parameter shape.
 * Provides accumulate, scale, clip, and zero methods.
 */
class GradientAccumulator {
  readonly data: Float64Array;
  readonly size: number;

  constructor(size: number) {
    this.size = size;
    this.data = new Float64Array(size);
  }

  /** Add gradients to buffer */
  accumulate(gradients: Float64Array, gradOff: number, len: number): void {
    for (let i = 0; i < len; i++) {
      this.data[i] += gradients[gradOff + i];
    }
  }

  /** Scale gradients by factor */
  scale(s: number): void {
    for (let i = 0; i < this.size; i++) {
      this.data[i] *= s;
    }
  }

  /** Clip gradients by L2 norm */
  clipByNorm(maxNorm: number): void {
    const norm = TensorOps.l2Norm(this.data, 0, this.size);
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (let i = 0; i < this.size; i++) {
        this.data[i] *= scale;
      }
    }
  }

  /** Zero all gradients */
  zero(): void {
    for (let i = 0; i < this.size; i++) {
      this.data[i] = 0;
    }
  }
}

/**
 * Stores first moment (m) and second moment (v) buffers for Adam.
 *
 * Adam update:
 *   m = β₁m + (1-β₁)g
 *   v = β₂v + (1-β₂)g²
 *   m̂ = m / (1 - β₁ᵗ)
 *   v̂ = v / (1 - β₂ᵗ)
 *   θ = θ - α * m̂ / (√v̂ + ε)
 */
class AdamState {
  readonly m: Float64Array;
  readonly v: Float64Array;
  readonly size: number;

  constructor(size: number) {
    this.size = size;
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
  }

  update(
    params: Float64Array,
    paramOff: number,
    grads: Float64Array,
    gradOff: number,
    lr: number,
    beta1: number,
    beta2: number,
    epsilon: number,
    t: number,
  ): void {
    const beta1Correction = 1 - Math.pow(beta1, t);
    const beta2Correction = 1 - Math.pow(beta2, t);
    const lrCorrected = lr * Math.sqrt(beta2Correction) / beta1Correction;

    for (let i = 0; i < this.size; i++) {
      const g = grads[gradOff + i];

      // Update biased first moment
      this.m[i] = beta1 * this.m[i] + (1 - beta1) * g;

      // Update biased second moment
      this.v[i] = beta2 * this.v[i] + (1 - beta2) * g * g;

      // Compute update (with bias correction baked into lr)
      params[paramOff + i] -= lrCorrected * this.m[i] /
        (Math.sqrt(this.v[i]) + epsilon);
    }
  }

  reset(): void {
    for (let i = 0; i < this.size; i++) {
      this.m[i] = 0;
      this.v[i] = 0;
    }
  }

  serialize(): { m: number[]; v: number[] } {
    return {
      m: Array.from(this.m),
      v: Array.from(this.v),
    };
  }

  deserialize(data: { m: number[]; v: number[] }): void {
    for (let i = 0; i < this.size; i++) {
      this.m[i] = data.m[i];
      this.v[i] = data.v[i];
    }
  }
}

/**
 * Manages collection of AdamState instances keyed by parameter name.
 * Applies learning rate, L2 decay, gradient clipping.
 */
class AdamOptimizer {
  private states: Map<string, AdamState> = new Map();
  private t: number = 0;
  private config: {
    lr: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    clipNorm: number;
    l2Lambda: number;
  };

  constructor(
    lr: number,
    beta1: number,
    beta2: number,
    epsilon: number,
    clipNorm: number,
    l2Lambda: number,
  ) {
    this.config = { lr, beta1, beta2, epsilon, clipNorm, l2Lambda };
  }

  /** Register a parameter for optimization */
  registerParameter(name: string, size: number): void {
    this.states.set(name, new AdamState(size));
  }

  /** Get Adam state for parameter */
  getState(name: string): AdamState | undefined {
    return this.states.get(name);
  }

  /**
   * Perform optimization step
   * @param parameters Map of parameter name to {params, grads} buffers
   */
  step(
    parameters: Map<
      string,
      { params: Float64Array; paramOff: number; grads: GradientAccumulator }
    >,
  ): void {
    this.t++;

    for (const [name, { params, paramOff, grads }] of parameters) {
      const state = this.states.get(name);
      if (!state) continue;

      // Clip gradients
      grads.clipByNorm(this.config.clipNorm);

      // Apply L2 regularization to gradients
      if (this.config.l2Lambda > 0) {
        for (let i = 0; i < state.size; i++) {
          grads.data[i] += 2 * this.config.l2Lambda * params[paramOff + i];
        }
      }

      // Adam update
      state.update(
        params,
        paramOff,
        grads.data,
        0,
        this.config.lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        this.t,
      );

      // Zero gradients
      grads.zero();
    }
  }

  getTimestep(): number {
    return this.t;
  }

  reset(): void {
    this.t = 0;
    for (const state of this.states.values()) {
      state.reset();
    }
  }

  serialize(): {
    t: number;
    states: { [key: string]: { m: number[]; v: number[] } };
  } {
    const states: { [key: string]: { m: number[]; v: number[] } } = {};
    for (const [name, state] of this.states) {
      states[name] = state.serialize();
    }
    return { t: this.t, states };
  }

  deserialize(
    data: {
      t: number;
      states: { [key: string]: { m: number[]; v: number[] } };
    },
  ): void {
    this.t = data.t;
    for (const [name, stateData] of Object.entries(data.states)) {
      const state = this.states.get(name);
      if (state) {
        state.deserialize(stateData);
      }
    }
  }
}

// ==================== PHASE 4: CONVOLUTION LAYERS ====================

/**
 * Precomputed index lookup table for causal dilated convolution.
 * Stores input indices for each output position. Built once, reused forever.
 */
class ConvIndexMap {
  private indices: Int32Array;
  private kernelSize: number;
  private seqLen: number;

  constructor(kernelSize: number, dilation: number, seqLen: number) {
    this.kernelSize = kernelSize;
    this.seqLen = seqLen;
    this.indices = new Int32Array(seqLen * kernelSize);

    // Precompute: for output position t and kernel position k,
    // input index = t - k * dilation (causal: only past inputs)
    // Store -1 if index < 0 (zero padding)
    for (let t = 0; t < seqLen; t++) {
      for (let k = 0; k < kernelSize; k++) {
        const inputIdx = t - k * dilation;
        this.indices[t * kernelSize + k] = inputIdx >= 0 ? inputIdx : -1;
      }
    }
  }

  /** Get input index for output position t and kernel position k */
  getInputIndex(t: number, k: number): number {
    return this.indices[t * this.kernelSize + k];
  }
}

/**
 * Holds weight and bias for causal 1D convolution.
 * Weight shape: [outChannels, inChannels, kernelSize]
 * Bias shape: [outChannels]
 */
class CausalConv1DParams {
  readonly weight: Float64Array;
  readonly bias: Float64Array;
  readonly weightGrad: GradientAccumulator;
  readonly biasGrad: GradientAccumulator;
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: number;

  constructor(inChannels: number, outChannels: number, kernelSize: number) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.weight = new Float64Array(outChannels * inChannels * kernelSize);
    this.bias = new Float64Array(outChannels);
    this.weightGrad = new GradientAccumulator(this.weight.length);
    this.biasGrad = new GradientAccumulator(outChannels);
  }

  /** Initialize with Xavier/He scaling */
  initialize(rng: RandomGenerator, scale: number): void {
    const fanIn = this.inChannels * this.kernelSize;
    const std = scale * Math.sqrt(2 / fanIn);
    for (let i = 0; i < this.weight.length; i++) {
      this.weight[i] = rng.truncatedGaussian(std);
    }
    for (let i = 0; i < this.bias.length; i++) {
      this.bias[i] = 0;
    }
  }

  /** Get weight at [outC, inC, k] */
  getWeight(outC: number, inC: number, k: number): number {
    return this.weight[(outC * this.inChannels + inC) * this.kernelSize + k];
  }

  /** Set weight at [outC, inC, k] */
  setWeight(outC: number, inC: number, k: number, value: number): void {
    this.weight[(outC * this.inChannels + inC) * this.kernelSize + k] = value;
  }

  zeroGradients(): void {
    this.weightGrad.zero();
    this.biasGrad.zero();
  }
}

/**
 * Causal dilated 1D convolution layer.
 * Forward: output[t, outC] = bias[outC] + Σ_k Σ_inC weight[outC, inC, k] * input[t - k*dilation, inC]
 */
class CausalConv1D {
  readonly params: CausalConv1DParams;
  private indexMap: ConvIndexMap;
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
    this.indexMap = new ConvIndexMap(kernelSize, dilation, maxSeqLen);
    this.dilation = dilation;
    this.maxSeqLen = maxSeqLen;
  }

  /** Initialize weights */
  initialize(rng: RandomGenerator, scale: number): void {
    this.params.initialize(rng, scale);
  }

  /**
   * Forward pass
   * @param output Shape [seqLen, outChannels]
   * @param input Shape [seqLen, inChannels]
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    const { inChannels, outChannels, kernelSize, weight, bias } = this.params;

    for (let t = 0; t < seqLen; t++) {
      for (let outC = 0; outC < outChannels; outC++) {
        let sum = bias[outC];

        for (let k = 0; k < kernelSize; k++) {
          const inIdx = this.indexMap.getInputIndex(t, k);
          if (inIdx >= 0) {
            for (let inC = 0; inC < inChannels; inC++) {
              const wIdx = (outC * inChannels + inC) * kernelSize + k;
              sum += weight[wIdx] * input[inOff + inIdx * inChannels + inC];
            }
          }
        }

        output[outOff + t * outChannels + outC] = sum;
      }
    }
  }

  /**
   * Backward pass - computes gradients for weights, bias, and input
   */
  backward(
    dInput: Float64Array | null,
    dInOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    const {
      inChannels,
      outChannels,
      kernelSize,
      weight,
      weightGrad,
      biasGrad,
    } = this.params;

    // Zero dInput if provided
    if (dInput !== null) {
      TensorOps.fill(dInput, dInOff, seqLen * inChannels, 0);
    }

    for (let t = 0; t < seqLen; t++) {
      for (let outC = 0; outC < outChannels; outC++) {
        const dOut = dOutput[dOutOff + t * outChannels + outC];

        // Bias gradient
        biasGrad.data[outC] += dOut;

        for (let k = 0; k < kernelSize; k++) {
          const inIdx = this.indexMap.getInputIndex(t, k);
          if (inIdx >= 0) {
            for (let inC = 0; inC < inChannels; inC++) {
              const wIdx = (outC * inChannels + inC) * kernelSize + k;
              const inVal = input[inOff + inIdx * inChannels + inC];

              // Weight gradient
              weightGrad.data[wIdx] += dOut * inVal;

              // Input gradient
              if (dInput !== null) {
                dInput[dInOff + inIdx * inChannels + inC] += dOut *
                  weight[wIdx];
              }
            }
          }
        }
      }
    }
  }

  zeroGradients(): void {
    this.params.zeroGradients();
  }

  getReceptiveFieldContribution(): number {
    return (this.params.kernelSize - 1) * this.dilation;
  }
}

/**
 * Pointwise 1x1 convolution for channel projection.
 * Weight shape: [outChannels, inChannels]
 * Bias shape: [outChannels]
 */
class Conv1x1 {
  readonly weight: Float64Array;
  readonly bias: Float64Array;
  readonly weightGrad: GradientAccumulator;
  readonly biasGrad: GradientAccumulator;
  readonly inChannels: number;
  readonly outChannels: number;

  constructor(inChannels: number, outChannels: number) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.weight = new Float64Array(outChannels * inChannels);
    this.bias = new Float64Array(outChannels);
    this.weightGrad = new GradientAccumulator(this.weight.length);
    this.biasGrad = new GradientAccumulator(outChannels);
  }

  initialize(rng: RandomGenerator, scale: number): void {
    const std = scale * Math.sqrt(2 / this.inChannels);
    for (let i = 0; i < this.weight.length; i++) {
      this.weight[i] = rng.truncatedGaussian(std);
    }
    for (let i = 0; i < this.bias.length; i++) {
      this.bias[i] = 0;
    }
  }

  /**
   * Forward: for each t, output[t] = weight @ input[t] + bias
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      TensorOps.matvec(
        output,
        outOff + t * this.outChannels,
        this.weight,
        0,
        input,
        inOff + t * this.inChannels,
        this.outChannels,
        this.inChannels,
      );
      // Add bias
      for (let c = 0; c < this.outChannels; c++) {
        output[outOff + t * this.outChannels + c] += this.bias[c];
      }
    }
  }

  /**
   * Backward pass
   */
  backward(
    dInput: Float64Array | null,
    dInOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    if (dInput !== null) {
      TensorOps.fill(dInput, dInOff, seqLen * this.inChannels, 0);
    }

    for (let t = 0; t < seqLen; t++) {
      // Bias gradient
      for (let c = 0; c < this.outChannels; c++) {
        this.biasGrad.data[c] += dOutput[dOutOff + t * this.outChannels + c];
      }

      // Weight gradient: dW += outer(dOut, input)
      TensorOps.outerAdd(
        this.weightGrad.data,
        0,
        dOutput,
        dOutOff + t * this.outChannels,
        input,
        inOff + t * this.inChannels,
        this.outChannels,
        this.inChannels,
      );

      // Input gradient: dInput = W^T @ dOut
      if (dInput !== null) {
        TensorOps.matvecT(
          dInput,
          dInOff + t * this.inChannels,
          this.weight,
          0,
          dOutput,
          dOutOff + t * this.outChannels,
          this.outChannels,
          this.inChannels,
        );
      }
    }
  }

  zeroGradients(): void {
    this.weightGrad.zero();
    this.biasGrad.zero();
  }
}

// ==================== PHASE 5: TCN BLOCKS ====================

/**
 * Preallocated binary mask buffer for dropout.
 */
class DropoutMask {
  private mask: Float64Array;
  private maxSize: number;

  constructor(maxSize: number) {
    this.maxSize = maxSize;
    this.mask = new Float64Array(maxSize);
  }

  /** Generate dropout mask */
  generate(rng: RandomGenerator, size: number, dropRate: number): void {
    if (dropRate === 0) {
      for (let i = 0; i < size; i++) {
        this.mask[i] = 1;
      }
    } else {
      const scale = 1 / (1 - dropRate);
      for (let i = 0; i < size; i++) {
        this.mask[i] = rng.nextFloat() >= dropRate ? scale : 0;
      }
    }
  }

  /** Apply mask to activations (forward) */
  applyForward(data: Float64Array, offset: number, size: number): void {
    for (let i = 0; i < size; i++) {
      data[offset + i] *= this.mask[i];
    }
  }

  /** Apply mask to gradients (backward) */
  applyBackward(grad: Float64Array, offset: number, size: number): void {
    for (let i = 0; i < size; i++) {
      grad[offset + i] *= this.mask[i];
    }
  }
}

/**
 * Layer normalization parameters: gamma (scale) and beta (shift)
 */
class LayerNormParams {
  readonly gamma: Float64Array;
  readonly beta: Float64Array;
  readonly gammaGrad: GradientAccumulator;
  readonly betaGrad: GradientAccumulator;
  readonly channels: number;

  constructor(channels: number) {
    this.channels = channels;
    this.gamma = new Float64Array(channels);
    this.beta = new Float64Array(channels);
    this.gammaGrad = new GradientAccumulator(channels);
    this.betaGrad = new GradientAccumulator(channels);

    // Initialize: gamma=1, beta=0
    for (let i = 0; i < channels; i++) {
      this.gamma[i] = 1;
      this.beta[i] = 0;
    }
  }

  zeroGradients(): void {
    this.gammaGrad.zero();
    this.betaGrad.zero();
  }
}

/**
 * Layer normalization operations
 *
 * Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
 */
class LayerNormOps {
  private static epsilon = 1e-5;

  /**
   * Forward pass - normalizes across channel dimension
   * @param output Output buffer [seqLen, channels]
   * @param input Input buffer [seqLen, channels]
   * @param stats Stored mean/invStd for backward [seqLen, 2]
   */
  static forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    gamma: Float64Array,
    beta: Float64Array,
    stats: Float64Array,
    statsOff: number,
    channels: number,
    seqLen: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const baseIn = inOff + t * channels;
      const baseOut = outOff + t * channels;
      const statsBase = statsOff + t * 2;

      // Compute mean
      let mean = 0;
      for (let c = 0; c < channels; c++) {
        mean += input[baseIn + c];
      }
      mean /= channels;

      // Compute variance
      let variance = 0;
      for (let c = 0; c < channels; c++) {
        const diff = input[baseIn + c] - mean;
        variance += diff * diff;
      }
      variance /= channels;

      const invStd = 1 / Math.sqrt(variance + LayerNormOps.epsilon);

      // Store for backward
      stats[statsBase] = mean;
      stats[statsBase + 1] = invStd;

      // Normalize, scale, shift
      for (let c = 0; c < channels; c++) {
        const normalized = (input[baseIn + c] - mean) * invStd;
        output[baseOut + c] = gamma[c] * normalized + beta[c];
      }
    }
  }

  /**
   * Backward pass
   */
  static backward(
    dInput: Float64Array,
    dInOff: number,
    dGamma: GradientAccumulator,
    dBeta: GradientAccumulator,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
    gamma: Float64Array,
    stats: Float64Array,
    statsOff: number,
    channels: number,
    seqLen: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const baseIn = inOff + t * channels;
      const baseDOut = dOutOff + t * channels;
      const baseDIn = dInOff + t * channels;
      const statsBase = statsOff + t * 2;

      const mean = stats[statsBase];
      const invStd = stats[statsBase + 1];

      // Compute gradients for gamma and beta
      for (let c = 0; c < channels; c++) {
        const normalized = (input[baseIn + c] - mean) * invStd;
        dGamma.data[c] += dOutput[baseDOut + c] * normalized;
        dBeta.data[c] += dOutput[baseDOut + c];
      }

      // Compute gradient for input (more complex due to mean/var dependencies)
      let sumDy = 0;
      let sumDyXhat = 0;
      for (let c = 0; c < channels; c++) {
        const dy = dOutput[baseDOut + c] * gamma[c];
        const xhat = (input[baseIn + c] - mean) * invStd;
        sumDy += dy;
        sumDyXhat += dy * xhat;
      }

      for (let c = 0; c < channels; c++) {
        const dy = dOutput[baseDOut + c] * gamma[c];
        const xhat = (input[baseIn + c] - mean) * invStd;
        dInput[baseDIn + c] = invStd *
          (dy - sumDy / channels - xhat * sumDyXhat / channels);
      }
    }
  }
}

/**
 * Context for storing activations during forward pass
 */
interface TCNBlockContext {
  preAct1: Float64Array;
  preAct2: Float64Array | null;
  residualInput: Float64Array | null;
  normStats: Float64Array | null;
  seqLen: number;
}

/**
 * Single residual TCN block.
 * Forward: conv1 -> activation -> [conv2 -> activation] -> [norm] -> [dropout] -> residual add
 */
class TCNBlock {
  private conv1: CausalConv1D;
  private conv2: CausalConv1D | null = null;
  private residualProj: Conv1x1 | null = null;
  private normParams: LayerNormParams | null = null;
  private dropoutMask: DropoutMask;

  readonly inChannels: number;
  readonly outChannels: number;
  private activation: "relu" | "gelu";
  private dropoutRate: number;
  private maxSeqLen: number;

  // Preallocated buffers for forward/backward
  private act1Buffer: Float64Array;
  private act2Buffer: Float64Array | null = null;
  private residualBuffer: Float64Array;
  private normBuffer: Float64Array | null = null;
  private normStatsBuffer: Float64Array | null = null;

  // Gradient buffers
  private dAct1Buffer: Float64Array;
  private dAct2Buffer: Float64Array | null = null;
  private dResidualBuffer: Float64Array;
  private dNormBuffer: Float64Array | null = null;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    dilation: number,
    useTwoLayers: boolean,
    activation: "relu" | "gelu",
    useNorm: boolean,
    dropoutRate: number,
    maxSeqLen: number,
  ) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.activation = activation;
    this.dropoutRate = dropoutRate;
    this.maxSeqLen = maxSeqLen;

    // Create conv1
    this.conv1 = new CausalConv1D(
      inChannels,
      outChannels,
      kernelSize,
      dilation,
      maxSeqLen,
    );

    // Create conv2 if two-layer block
    if (useTwoLayers) {
      this.conv2 = new CausalConv1D(
        outChannels,
        outChannels,
        kernelSize,
        dilation,
        maxSeqLen,
      );
      this.act2Buffer = new Float64Array(maxSeqLen * outChannels);
      this.dAct2Buffer = new Float64Array(maxSeqLen * outChannels);
    }

    // Create residual projection if dimensions differ
    if (inChannels !== outChannels) {
      this.residualProj = new Conv1x1(inChannels, outChannels);
    }

    // Create layer norm if enabled
    if (useNorm) {
      this.normParams = new LayerNormParams(outChannels);
      this.normBuffer = new Float64Array(maxSeqLen * outChannels);
      this.normStatsBuffer = new Float64Array(maxSeqLen * 2);
      this.dNormBuffer = new Float64Array(maxSeqLen * outChannels);
    }

    // Create dropout mask
    this.dropoutMask = new DropoutMask(maxSeqLen * outChannels);

    // Preallocate activation buffers
    this.act1Buffer = new Float64Array(maxSeqLen * outChannels);
    this.residualBuffer = new Float64Array(maxSeqLen * outChannels);
    this.dAct1Buffer = new Float64Array(maxSeqLen * outChannels);
    this.dResidualBuffer = new Float64Array(maxSeqLen * outChannels);
  }

  initialize(rng: RandomGenerator, scale: number): void {
    this.conv1.initialize(rng, scale);
    if (this.conv2) this.conv2.initialize(rng, scale);
    if (this.residualProj) this.residualProj.initialize(rng, scale);
  }

  /**
   * Forward pass
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
    training: boolean,
    rng: RandomGenerator,
    context: TCNBlockContext | null,
  ): void {
    const actSize = seqLen * this.outChannels;

    // Conv1 -> pre-activation
    this.conv1.forward(this.act1Buffer, 0, input, inOff, seqLen);

    // Store pre-activation for backward if training
    if (context) {
      TensorOps.copy(context.preAct1, 0, this.act1Buffer, 0, actSize);
    }

    // Activation
    if (this.activation === "relu") {
      ActivationOps.reluInPlace(this.act1Buffer, 0, actSize);
    } else {
      ActivationOps.geluInPlace(this.act1Buffer, 0, actSize);
    }

    // Optional second conv layer
    let currentAct = this.act1Buffer;
    if (this.conv2 && this.act2Buffer) {
      this.conv2.forward(this.act2Buffer, 0, this.act1Buffer, 0, seqLen);

      if (context && context.preAct2) {
        TensorOps.copy(context.preAct2, 0, this.act2Buffer, 0, actSize);
      }

      if (this.activation === "relu") {
        ActivationOps.reluInPlace(this.act2Buffer, 0, actSize);
      } else {
        ActivationOps.geluInPlace(this.act2Buffer, 0, actSize);
      }

      currentAct = this.act2Buffer;
    }

    // Optional layer norm
    if (this.normParams && this.normBuffer && this.normStatsBuffer) {
      LayerNormOps.forward(
        this.normBuffer,
        0,
        currentAct,
        0,
        this.normParams.gamma,
        this.normParams.beta,
        this.normStatsBuffer,
        0,
        this.outChannels,
        seqLen,
      );

      if (context && context.normStats) {
        TensorOps.copy(
          context.normStats,
          0,
          this.normStatsBuffer,
          0,
          seqLen * 2,
        );
      }

      currentAct = this.normBuffer;
    }

    // Optional dropout (training only)
    if (training && this.dropoutRate > 0) {
      this.dropoutMask.generate(rng, actSize, this.dropoutRate);
      this.dropoutMask.applyForward(currentAct, 0, actSize);
    }

    // Compute residual
    if (this.residualProj) {
      this.residualProj.forward(this.residualBuffer, 0, input, inOff, seqLen);
    } else {
      // If channels match, copy input as residual
      if (this.inChannels === this.outChannels) {
        TensorOps.copy(this.residualBuffer, 0, input, inOff, actSize);
      }
    }

    if (context && context.residualInput) {
      TensorOps.copy(
        context.residualInput,
        0,
        input,
        inOff,
        seqLen * this.inChannels,
      );
    }

    // Output = activation + residual
    TensorOps.add(
      output,
      outOff,
      currentAct,
      0,
      this.residualBuffer,
      0,
      actSize,
    );
  }

  /**
   * Backward pass
   */
  backward(
    dInput: Float64Array | null,
    dInOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
    context: TCNBlockContext,
    training: boolean,
  ): void {
    const seqLen = context.seqLen;
    const actSize = seqLen * this.outChannels;

    // dOutput flows to both activation path and residual path
    TensorOps.copy(this.dResidualBuffer, 0, dOutput, dOutOff, actSize);

    let currentDOut = dOutput;
    let currentDOutOff = dOutOff;

    // Backprop through dropout if applied
    if (training && this.dropoutRate > 0) {
      TensorOps.copy(this.dAct1Buffer, 0, dOutput, dOutOff, actSize);
      this.dropoutMask.applyBackward(this.dAct1Buffer, 0, actSize);
      currentDOut = this.dAct1Buffer;
      currentDOutOff = 0;
    }

    // Backprop through layer norm
    if (this.normParams && this.dNormBuffer && context.normStats) {
      const act = this.conv2 ? this.act2Buffer! : this.act1Buffer;
      LayerNormOps.backward(
        this.dNormBuffer,
        0,
        this.normParams.gammaGrad,
        this.normParams.betaGrad,
        currentDOut,
        currentDOutOff,
        act,
        0,
        this.normParams.gamma,
        context.normStats,
        0,
        this.outChannels,
        seqLen,
      );
      currentDOut = this.dNormBuffer;
      currentDOutOff = 0;
    }

    // Backprop through second conv + activation
    if (this.conv2 && this.dAct2Buffer && context.preAct2) {
      // Activation backward
      if (this.activation === "relu") {
        ActivationOps.reluBackward(
          this.dAct2Buffer,
          0,
          currentDOut,
          currentDOutOff,
          context.preAct2,
          0,
          actSize,
        );
      } else {
        ActivationOps.geluBackward(
          this.dAct2Buffer,
          0,
          currentDOut,
          currentDOutOff,
          context.preAct2,
          0,
          actSize,
        );
      }

      // Conv2 backward
      this.conv2.backward(
        this.dAct1Buffer,
        0,
        this.dAct2Buffer,
        0,
        this.act1Buffer,
        0,
        seqLen,
      );
      currentDOut = this.dAct1Buffer;
      currentDOutOff = 0;
    }

    // Backprop through first conv + activation
    if (this.activation === "relu") {
      ActivationOps.reluBackward(
        this.dAct1Buffer,
        0,
        currentDOut,
        currentDOutOff,
        context.preAct1,
        0,
        actSize,
      );
    } else {
      ActivationOps.geluBackward(
        this.dAct1Buffer,
        0,
        currentDOut,
        currentDOutOff,
        context.preAct1,
        0,
        actSize,
      );
    }

    // Conv1 backward
    const dInTemp = dInput !== null
      ? new Float64Array(seqLen * this.inChannels)
      : null;
    this.conv1.backward(dInTemp, 0, this.dAct1Buffer, 0, input, inOff, seqLen);

    // Backprop through residual path
    if (this.residualProj && context.residualInput) {
      const dResidualInput = new Float64Array(seqLen * this.inChannels);
      this.residualProj.backward(
        dResidualInput,
        0,
        this.dResidualBuffer,
        0,
        context.residualInput,
        0,
        seqLen,
      );

      // Add to input gradient
      if (dInput !== null && dInTemp !== null) {
        TensorOps.add(
          dInput,
          dInOff,
          dInTemp,
          0,
          dResidualInput,
          0,
          seqLen * this.inChannels,
        );
      }
    } else if (dInput !== null && dInTemp !== null) {
      // Add residual gradient directly (same dimension)
      for (let i = 0; i < actSize; i++) {
        dInput[dInOff + i] = dInTemp[i] + this.dResidualBuffer[i];
      }
    }
  }

  zeroGradients(): void {
    this.conv1.zeroGradients();
    if (this.conv2) this.conv2.zeroGradients();
    if (this.residualProj) this.residualProj.zeroGradients();
    if (this.normParams) this.normParams.zeroGradients();
  }

  /** Collect all parameters for optimizer */
  getParameters(): Map<
    string,
    { params: Float64Array; paramOff: number; grads: GradientAccumulator }
  > {
    const params = new Map<
      string,
      { params: Float64Array; paramOff: number; grads: GradientAccumulator }
    >();
    params.set("conv1_weight", {
      params: this.conv1.params.weight,
      paramOff: 0,
      grads: this.conv1.params.weightGrad,
    });
    params.set("conv1_bias", {
      params: this.conv1.params.bias,
      paramOff: 0,
      grads: this.conv1.params.biasGrad,
    });

    if (this.conv2) {
      params.set("conv2_weight", {
        params: this.conv2.params.weight,
        paramOff: 0,
        grads: this.conv2.params.weightGrad,
      });
      params.set("conv2_bias", {
        params: this.conv2.params.bias,
        paramOff: 0,
        grads: this.conv2.params.biasGrad,
      });
    }

    if (this.residualProj) {
      params.set("proj_weight", {
        params: this.residualProj.weight,
        paramOff: 0,
        grads: this.residualProj.weightGrad,
      });
      params.set("proj_bias", {
        params: this.residualProj.bias,
        paramOff: 0,
        grads: this.residualProj.biasGrad,
      });
    }

    if (this.normParams) {
      params.set("norm_gamma", {
        params: this.normParams.gamma,
        paramOff: 0,
        grads: this.normParams.gammaGrad,
      });
      params.set("norm_beta", {
        params: this.normParams.beta,
        paramOff: 0,
        grads: this.normParams.betaGrad,
      });
    }

    return params;
  }

  getReceptiveFieldContribution(): number {
    let rf = this.conv1.getReceptiveFieldContribution();
    if (this.conv2) {
      rf += this.conv2.getReceptiveFieldContribution();
    }
    return rf;
  }

  createContext(seqLen: number): TCNBlockContext {
    return {
      preAct1: new Float64Array(seqLen * this.outChannels),
      preAct2: this.conv2 ? new Float64Array(seqLen * this.outChannels) : null,
      residualInput: this.residualProj
        ? new Float64Array(seqLen * this.inChannels)
        : null,
      normStats: this.normParams ? new Float64Array(seqLen * 2) : null,
      seqLen,
    };
  }
}

/**
 * Stack of TCNBlock instances with computed dilation schedule.
 */
class TCNBackbone {
  private blocks: TCNBlock[] = [];
  private dilations: number[] = [];
  private receptiveField: number;
  readonly hiddenChannels: number;

  constructor(
    nFeatures: number,
    hiddenChannels: number,
    nBlocks: number,
    kernelSize: number,
    dilationBase: number,
    useTwoLayerBlock: boolean,
    activation: "relu" | "gelu",
    useLayerNorm: boolean,
    dropoutRate: number,
    maxSeqLen: number,
  ) {
    this.hiddenChannels = hiddenChannels;

    // Compute dilation schedule: powers of dilationBase
    for (let i = 0; i < nBlocks; i++) {
      this.dilations.push(Math.pow(dilationBase, i));
    }

    // Create blocks
    for (let i = 0; i < nBlocks; i++) {
      const inC = i === 0 ? nFeatures : hiddenChannels;
      const outC = hiddenChannels;
      const block = new TCNBlock(
        inC,
        outC,
        kernelSize,
        this.dilations[i],
        useTwoLayerBlock,
        activation,
        useLayerNorm,
        dropoutRate,
        maxSeqLen,
      );
      this.blocks.push(block);
    }

    // Compute receptive field
    this.receptiveField = 1;
    for (const block of this.blocks) {
      this.receptiveField += block.getReceptiveFieldContribution();
    }
  }

  initialize(rng: RandomGenerator, scale: number): void {
    for (const block of this.blocks) {
      block.initialize(rng, scale);
    }
  }

  /**
   * Forward pass through all blocks
   * @returns Output shape [seqLen, hiddenChannels]
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
    training: boolean,
    rng: RandomGenerator,
    contexts: TCNBlockContext[] | null,
  ): void {
    let currentInput = input;
    let currentInOff = inOff;

    // Temporary buffers for intermediate outputs
    const tempBuffer1 = new Float64Array(seqLen * this.hiddenChannels);
    const tempBuffer2 = new Float64Array(seqLen * this.hiddenChannels);
    let useBuffer1 = true;

    for (let i = 0; i < this.blocks.length; i++) {
      const block = this.blocks[i];
      const isLast = i === this.blocks.length - 1;
      const context = contexts ? contexts[i] : null;

      let outBuffer: Float64Array;
      let outBufferOff: number;

      if (isLast) {
        outBuffer = output;
        outBufferOff = outOff;
      } else {
        outBuffer = useBuffer1 ? tempBuffer1 : tempBuffer2;
        outBufferOff = 0;
      }

      block.forward(
        outBuffer,
        outBufferOff,
        currentInput,
        currentInOff,
        seqLen,
        training,
        rng,
        context,
      );

      if (!isLast) {
        currentInput = outBuffer;
        currentInOff = 0;
        useBuffer1 = !useBuffer1;
      }
    }
  }

  /**
   * Backward pass through all blocks
   */
  backward(
    dInput: Float64Array | null,
    dInOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
    contexts: TCNBlockContext[],
    training: boolean,
  ): void {
    // Store intermediate inputs for backward
    const intermediates: { data: Float64Array; offset: number }[] = [];

    // Forward pass to get intermediates (needed for backward)
    let currentInput = input;
    let currentInOff = inOff;
    intermediates.push({ data: input, offset: inOff });

    const tempBuffers: Float64Array[] = [];
    for (let i = 0; i < this.blocks.length - 1; i++) {
      const buffer = new Float64Array(seqLen * this.hiddenChannels);
      tempBuffers.push(buffer);
    }

    // Recompute forward to get intermediates
    for (let i = 0; i < this.blocks.length - 1; i++) {
      const block = this.blocks[i];
      block.forward(
        tempBuffers[i],
        0,
        currentInput,
        currentInOff,
        seqLen,
        false,
        null as any,
        null,
      );
      currentInput = tempBuffers[i];
      currentInOff = 0;
      intermediates.push({ data: tempBuffers[i], offset: 0 });
    }

    // Backward through blocks in reverse
    let currentDOut = dOutput;
    let currentDOutOff = dOutOff;
    const dOutTemp = new Float64Array(seqLen * this.hiddenChannels);

    for (let i = this.blocks.length - 1; i >= 0; i--) {
      const block = this.blocks[i];
      const inputData = intermediates[i];
      const isFirst = i === 0;

      const dIn = isFirst ? dInput : dOutTemp;
      const dInOff2 = isFirst ? dInOff : 0;

      block.backward(
        dIn,
        dInOff2,
        currentDOut,
        currentDOutOff,
        inputData.data,
        inputData.offset,
        contexts[i],
        training,
      );

      if (!isFirst) {
        currentDOut = dOutTemp;
        currentDOutOff = 0;
      }
    }
  }

  zeroGradients(): void {
    for (const block of this.blocks) {
      block.zeroGradients();
    }
  }

  /** Collect all parameters for optimizer */
  getParameters(
    prefix: string,
  ): Map<
    string,
    { params: Float64Array; paramOff: number; grads: GradientAccumulator }
  > {
    const allParams = new Map<
      string,
      { params: Float64Array; paramOff: number; grads: GradientAccumulator }
    >();
    for (let i = 0; i < this.blocks.length; i++) {
      const blockParams = this.blocks[i].getParameters();
      for (const [name, param] of blockParams) {
        allParams.set(`${prefix}block${i}_${name}`, param);
      }
    }
    return allParams;
  }

  getReceptiveField(): number {
    return this.receptiveField;
  }

  getBlockCount(): number {
    return this.blocks.length;
  }

  createContexts(seqLen: number): TCNBlockContext[] {
    return this.blocks.map((block) => block.createContext(seqLen));
  }
}

// ==================== PHASE 6: OUTPUT HEAD ====================

/**
 * Fully connected layer for final prediction.
 * Weight shape: [outDim, inDim]
 * Bias shape: [outDim]
 */
class LinearLayer {
  readonly weight: Float64Array;
  readonly bias: Float64Array;
  readonly weightGrad: GradientAccumulator;
  readonly biasGrad: GradientAccumulator;
  readonly inDim: number;
  readonly outDim: number;

  constructor(inDim: number, outDim: number) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.weight = new Float64Array(outDim * inDim);
    this.bias = new Float64Array(outDim);
    this.weightGrad = new GradientAccumulator(this.weight.length);
    this.biasGrad = new GradientAccumulator(outDim);
  }

  initialize(rng: RandomGenerator, scale: number): void {
    const std = scale * Math.sqrt(2 / this.inDim);
    for (let i = 0; i < this.weight.length; i++) {
      this.weight[i] = rng.truncatedGaussian(std);
    }
    for (let i = 0; i < this.bias.length; i++) {
      this.bias[i] = 0;
    }
  }

  /**
   * Forward: output = weight @ input + bias
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
  ): void {
    TensorOps.matvec(
      output,
      outOff,
      this.weight,
      0,
      input,
      inOff,
      this.outDim,
      this.inDim,
    );
    for (let i = 0; i < this.outDim; i++) {
      output[outOff + i] += this.bias[i];
    }
  }

  /**
   * Backward pass
   */
  backward(
    dInput: Float64Array | null,
    dInOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    input: Float64Array,
    inOff: number,
  ): void {
    // Bias gradient
    for (let i = 0; i < this.outDim; i++) {
      this.biasGrad.data[i] += dOutput[dOutOff + i];
    }

    // Weight gradient: dW += outer(dOutput, input)
    TensorOps.outerAdd(
      this.weightGrad.data,
      0,
      dOutput,
      dOutOff,
      input,
      inOff,
      this.outDim,
      this.inDim,
    );

    // Input gradient: dInput = W^T @ dOutput
    if (dInput !== null) {
      TensorOps.matvecT(
        dInput,
        dInOff,
        this.weight,
        0,
        dOutput,
        dOutOff,
        this.outDim,
        this.inDim,
      );
    }
  }

  zeroGradients(): void {
    this.weightGrad.zero();
    this.biasGrad.zero();
  }
}

/**
 * Output projection from backbone hidden state to predictions.
 * If direct: single LinearLayer mapping last timestep to [nTargets * maxFutureSteps]
 * If recursive: LinearLayer mapping to [nTargets], used iteratively
 */
class MultiHorizonHead {
  private linearLayer: LinearLayer;
  readonly hiddenChannels: number;
  readonly nTargets: number;
  readonly maxFutureSteps: number;
  readonly useDirect: boolean;

  constructor(
    hiddenChannels: number,
    nTargets: number,
    maxFutureSteps: number,
    useDirect: boolean,
  ) {
    this.hiddenChannels = hiddenChannels;
    this.nTargets = nTargets;
    this.maxFutureSteps = maxFutureSteps;
    this.useDirect = useDirect;

    const outDim = useDirect ? nTargets * maxFutureSteps : nTargets;
    this.linearLayer = new LinearLayer(hiddenChannels, outDim);
  }

  initialize(rng: RandomGenerator, scale: number): void {
    this.linearLayer.initialize(rng, scale);
  }

  /**
   * Forward pass
   * @param hidden Last timestep hidden state [hiddenChannels]
   * @param output Output buffer [maxFutureSteps, nTargets] for direct or [nTargets] for recursive
   */
  forward(
    output: Float64Array,
    outOff: number,
    hidden: Float64Array,
    hiddenOff: number,
  ): void {
    this.linearLayer.forward(output, outOff, hidden, hiddenOff);
  }

  backward(
    dHidden: Float64Array | null,
    dHiddenOff: number,
    dOutput: Float64Array,
    dOutOff: number,
    hidden: Float64Array,
    hiddenOff: number,
  ): void {
    this.linearLayer.backward(
      dHidden,
      dHiddenOff,
      dOutput,
      dOutOff,
      hidden,
      hiddenOff,
    );
  }

  zeroGradients(): void {
    this.linearLayer.zeroGradients();
  }

  getParameters(
    prefix: string,
  ): Map<
    string,
    { params: Float64Array; paramOff: number; grads: GradientAccumulator }
  > {
    const params = new Map<
      string,
      { params: Float64Array; paramOff: number; grads: GradientAccumulator }
    >();
    params.set(`${prefix}weight`, {
      params: this.linearLayer.weight,
      paramOff: 0,
      grads: this.linearLayer.weightGrad,
    });
    params.set(`${prefix}bias`, {
      params: this.linearLayer.bias,
      paramOff: 0,
      grads: this.linearLayer.biasGrad,
    });
    return params;
  }
}

// ==================== PHASE 7: MODEL ASSEMBLY ====================

/**
 * Context for storing data during forward pass (needed for backward)
 */
interface ForwardContext {
  backboneContexts: TCNBlockContext[];
  inputCopy: Float64Array;
  lastHidden: Float64Array;
  seqLen: number;
}

/**
 * Core TCN model combining backbone and head
 */
class TCNModel {
  private backbone: TCNBackbone;
  private head: MultiHorizonHead;
  private rng: RandomGenerator;

  readonly config: {
    nFeatures: number;
    nTargets: number;
    hiddenChannels: number;
    maxSeqLen: number;
    maxFutureSteps: number;
    useDirect: boolean;
  };

  // Preallocated buffers
  private backboneOutput: Float64Array;
  private headOutput: Float64Array;

  constructor(
    nFeatures: number,
    nTargets: number,
    hiddenChannels: number,
    nBlocks: number,
    kernelSize: number,
    dilationBase: number,
    useTwoLayerBlock: boolean,
    activation: "relu" | "gelu",
    useLayerNorm: boolean,
    dropoutRate: number,
    maxSeqLen: number,
    maxFutureSteps: number,
    useDirect: boolean,
    seed: number,
    weightInitScale: number,
  ) {
    this.config = {
      nFeatures,
      nTargets,
      hiddenChannels,
      maxSeqLen,
      maxFutureSteps,
      useDirect,
    };

    this.rng = new RandomGenerator(seed);

    this.backbone = new TCNBackbone(
      nFeatures,
      hiddenChannels,
      nBlocks,
      kernelSize,
      dilationBase,
      useTwoLayerBlock,
      activation,
      useLayerNorm,
      dropoutRate,
      maxSeqLen,
    );

    this.head = new MultiHorizonHead(
      hiddenChannels,
      nTargets,
      maxFutureSteps,
      useDirect,
    );

    // Initialize
    this.backbone.initialize(this.rng, weightInitScale);
    this.head.initialize(this.rng, weightInitScale);

    // Preallocate buffers
    this.backboneOutput = new Float64Array(maxSeqLen * hiddenChannels);
    const headOutDim = useDirect ? nTargets * maxFutureSteps : nTargets;
    this.headOutput = new Float64Array(headOutDim);
  }

  /**
   * Forward pass (inference only)
   */
  forward(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
  ): void {
    // Backbone
    this.backbone.forward(
      this.backboneOutput,
      0,
      input,
      inOff,
      seqLen,
      false,
      this.rng,
      null,
    );

    // Get last timestep hidden state
    const lastHiddenOff = (seqLen - 1) * this.config.hiddenChannels;

    // Head
    this.head.forward(output, outOff, this.backboneOutput, lastHiddenOff);
  }

  /**
   * Forward pass with context recording (for training)
   */
  forwardTrain(
    output: Float64Array,
    outOff: number,
    input: Float64Array,
    inOff: number,
    seqLen: number,
    context: ForwardContext,
  ): void {
    // Store input copy for backward
    TensorOps.copy(
      context.inputCopy,
      0,
      input,
      inOff,
      seqLen * this.config.nFeatures,
    );

    // Backbone with context recording
    this.backbone.forward(
      this.backboneOutput,
      0,
      input,
      inOff,
      seqLen,
      true,
      this.rng,
      context.backboneContexts,
    );

    // Store last hidden state
    const lastHiddenOff = (seqLen - 1) * this.config.hiddenChannels;
    TensorOps.copy(
      context.lastHidden,
      0,
      this.backboneOutput,
      lastHiddenOff,
      this.config.hiddenChannels,
    );

    context.seqLen = seqLen;

    // Head
    this.head.forward(output, outOff, this.backboneOutput, lastHiddenOff);
  }

  /**
   * Backward pass
   */
  backward(
    dOutput: Float64Array,
    dOutOff: number,
    context: ForwardContext,
  ): void {
    const { hiddenChannels, nFeatures } = this.config;
    const seqLen = context.seqLen;

    // Head backward
    const dHidden = new Float64Array(hiddenChannels);
    this.head.backward(dHidden, 0, dOutput, dOutOff, context.lastHidden, 0);

    // Create dBackboneOutput (only last timestep has gradient from head)
    const dBackboneOutput = new Float64Array(seqLen * hiddenChannels);
    const lastHiddenOff = (seqLen - 1) * hiddenChannels;
    TensorOps.copy(dBackboneOutput, lastHiddenOff, dHidden, 0, hiddenChannels);

    // Backbone backward (don't need input gradient)
    this.backbone.backward(
      null,
      0,
      dBackboneOutput,
      0,
      context.inputCopy,
      0,
      seqLen,
      context.backboneContexts,
      true,
    );
  }

  zeroGradients(): void {
    this.backbone.zeroGradients();
    this.head.zeroGradients();
  }

  /** Collect all parameters */
  getAllParameters(): Map<
    string,
    { params: Float64Array; paramOff: number; grads: GradientAccumulator }
  > {
    const allParams = new Map<
      string,
      { params: Float64Array; paramOff: number; grads: GradientAccumulator }
    >();

    for (const [name, param] of this.backbone.getParameters("backbone_")) {
      allParams.set(name, param);
    }

    for (const [name, param] of this.head.getParameters("head_")) {
      allParams.set(name, param);
    }

    return allParams;
  }

  getParameterCount(): number {
    let count = 0;
    for (const [, param] of this.getAllParameters()) {
      count += param.params.length;
    }
    return count;
  }

  getReceptiveField(): number {
    return this.backbone.getReceptiveField();
  }

  getBlockCount(): number {
    return this.backbone.getBlockCount();
  }

  createContext(seqLen: number): ForwardContext {
    return {
      backboneContexts: this.backbone.createContexts(seqLen),
      inputCopy: new Float64Array(seqLen * this.config.nFeatures),
      lastHidden: new Float64Array(this.config.hiddenChannels),
      seqLen,
    };
  }

  /** Re-initialize all parameters */
  reinitialize(seed: number, scale: number): void {
    this.rng = new RandomGenerator(seed);
    this.backbone.initialize(this.rng, scale);
    this.head.initialize(this.rng, scale);
  }

  getRng(): RandomGenerator {
    return this.rng;
  }
}

// ==================== PHASE 8: TRAINING UTILITIES ====================

/**
 * Fixed-size circular buffer for sequence history.
 * Stores [maxSeqLen, nFeatures] data.
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

  /** Push new timestep to buffer */
  push(features: Float64Array, offset: number): void {
    const targetOff = this.head * this.nFeatures;
    TensorOps.copy(this.data, targetOff, features, offset, this.nFeatures);
    this.head = (this.head + 1) % this.maxLen;
    if (this.count < this.maxLen) this.count++;
  }

  /** Push from number array */
  pushArray(features: number[]): void {
    const targetOff = this.head * this.nFeatures;
    for (let i = 0; i < this.nFeatures; i++) {
      this.data[targetOff + i] = features[i];
    }
    this.head = (this.head + 1) % this.maxLen;
    if (this.count < this.maxLen) this.count++;
  }

  /** Get length of buffered sequence */
  getLength(): number {
    return this.count;
  }

  /**
   * Copy last `len` timesteps to output buffer (handles wraparound)
   */
  getWindow(output: Float64Array, outOff: number, len: number): void {
    const actualLen = Math.min(len, this.count);
    let readPos = (this.head - actualLen + this.maxLen) % this.maxLen;

    for (let t = 0; t < actualLen; t++) {
      const srcOff = readPos * this.nFeatures;
      const dstOff = outOff + t * this.nFeatures;
      TensorOps.copy(output, dstOff, this.data, srcOff, this.nFeatures);
      readPos = (readPos + 1) % this.maxLen;
    }
  }

  /** Get latest entry */
  getLatest(output: Float64Array, outOff: number): void {
    const readPos = (this.head - 1 + this.maxLen) % this.maxLen;
    TensorOps.copy(
      output,
      outOff,
      this.data,
      readPos * this.nFeatures,
      this.nFeatures,
    );
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
    TensorOps.fill(this.data, 0, this.data.length, 0);
  }

  serialize(): { data: number[]; head: number; count: number } {
    return {
      data: Array.from(this.data),
      head: this.head,
      count: this.count,
    };
  }

  deserialize(state: { data: number[]; head: number; count: number }): void {
    for (let i = 0; i < state.data.length; i++) {
      this.data[i] = state.data[i];
    }
    this.head = state.head;
    this.count = state.count;
  }
}

/**
 * Circular buffer storing recent prediction residuals for uncertainty estimation.
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

  /** Add residual (actual - predicted) */
  addResidual(
    predicted: Float64Array,
    predOff: number,
    actual: Float64Array,
    actualOff: number,
  ): void {
    const targetOff = this.head * this.nTargets;
    for (let i = 0; i < this.nTargets; i++) {
      this.residuals[targetOff + i] = actual[actualOff + i] -
        predicted[predOff + i];
    }
    this.head = (this.head + 1) % this.windowSize;
    if (this.count < this.windowSize) this.count++;
  }

  /** Get mean and std per target */
  getStats(): { mean: number[]; std: number[] } {
    const mean = new Array<number>(this.nTargets).fill(0);
    const std = new Array<number>(this.nTargets).fill(0);

    if (this.count === 0) return { mean, std };

    // Compute mean
    for (let i = 0; i < this.count; i++) {
      for (let t = 0; t < this.nTargets; t++) {
        mean[t] += this.residuals[i * this.nTargets + t];
      }
    }
    for (let t = 0; t < this.nTargets; t++) {
      mean[t] /= this.count;
    }

    // Compute std
    for (let i = 0; i < this.count; i++) {
      for (let t = 0; t < this.nTargets; t++) {
        const diff = this.residuals[i * this.nTargets + t] - mean[t];
        std[t] += diff * diff;
      }
    }
    for (let t = 0; t < this.nTargets; t++) {
      std[t] = Math.sqrt(std[t] / Math.max(1, this.count - 1) + 1e-8);
    }

    return { mean, std };
  }

  /** Get uncertainty bounds */
  getUncertaintyBounds(
    predictions: Float64Array,
    predOff: number,
    nSteps: number,
    multiplier: number,
  ): { lower: number[][]; upper: number[][] } {
    const stats = this.getStats();
    const lower: number[][] = [];
    const upper: number[][] = [];

    for (let s = 0; s < nSteps; s++) {
      const lowerStep: number[] = [];
      const upperStep: number[] = [];
      for (let t = 0; t < this.nTargets; t++) {
        const pred = predictions[predOff + s * this.nTargets + t];
        const halfWidth = multiplier * stats.std[t] * Math.sqrt(s + 1);
        lowerStep.push(pred - halfWidth);
        upperStep.push(pred + halfWidth);
      }
      lower.push(lowerStep);
      upper.push(upperStep);
    }

    return { lower, upper };
  }

  reset(): void {
    this.head = 0;
    this.count = 0;
    TensorOps.fill(this.residuals, 0, this.residuals.length, 0);
  }

  getCount(): number {
    return this.count;
  }

  serialize(): { residuals: number[]; head: number; count: number } {
    return {
      residuals: Array.from(this.residuals),
      head: this.head,
      count: this.count,
    };
  }

  deserialize(
    state: { residuals: number[]; head: number; count: number },
  ): void {
    for (let i = 0; i < state.residuals.length; i++) {
      this.residuals[i] = state.residuals[i];
    }
    this.head = state.head;
    this.count = state.count;
  }
}

/**
 * Single bucket in ADWIN data structure
 */
class ADWINBucket {
  total: number = 0;
  variance: number = 0;
  count: number = 0;

  add(value: number): void {
    this.count++;
    const delta = value - this.total / Math.max(1, this.count - 1);
    this.total += value;
    const delta2 = value - this.total / this.count;
    this.variance += delta * delta2;
  }

  merge(other: ADWINBucket): void {
    const n1 = this.count;
    const n2 = other.count;
    if (n1 === 0 && n2 === 0) return;

    const mean1 = n1 > 0 ? this.total / n1 : 0;
    const mean2 = n2 > 0 ? other.total / n2 : 0;
    const newCount = n1 + n2;
    const newTotal = this.total + other.total;
    const newMean = newTotal / newCount;

    // Combined variance using parallel algorithm
    const delta = mean2 - mean1;
    const newVariance = this.variance + other.variance +
      delta * delta * n1 * n2 / newCount;

    this.count = newCount;
    this.total = newTotal;
    this.variance = newVariance;
  }

  reset(): void {
    this.total = 0;
    this.variance = 0;
    this.count = 0;
  }

  getMean(): number {
    return this.count > 0 ? this.total / this.count : 0;
  }
}

/**
 * Adaptive windowing algorithm for concept drift detection.
 * Maintains hierarchical buckets with exponential compression.
 */
class ADWINDetector {
  private buckets: ADWINBucket[][] = [];
  private delta: number;
  private maxBuckets: number;
  private totalCount: number = 0;
  private lastChangeDetected: boolean = false;

  constructor(delta: number = 0.002, maxBuckets: number = 64) {
    this.delta = delta;
    this.maxBuckets = maxBuckets;
    this.buckets = [];
    for (let i = 0; i < Math.ceil(Math.log2(maxBuckets)) + 1; i++) {
      this.buckets.push([]);
    }
  }

  /** Update with new value, returns whether drift was detected */
  update(value: number): boolean {
    // Add to level 0
    const newBucket = new ADWINBucket();
    newBucket.add(value);

    if (this.buckets[0].length < this.maxBuckets) {
      this.buckets[0].push(newBucket);
    } else {
      // Merge oldest buckets if at capacity
      this.compress();
      this.buckets[0].push(newBucket);
    }

    this.totalCount++;

    // Check for drift
    this.lastChangeDetected = this.checkDrift();

    return this.lastChangeDetected;
  }

  private compress(): void {
    for (let level = 0; level < this.buckets.length - 1; level++) {
      if (this.buckets[level].length >= 2) {
        const b1 = this.buckets[level].shift()!;
        const b2 = this.buckets[level].shift()!;
        b1.merge(b2);

        if (this.buckets[level + 1].length < this.maxBuckets) {
          this.buckets[level + 1].push(b1);
        }
        break;
      }
    }
  }

  private checkDrift(): boolean {
    // Get all data points
    let totalSum = 0;
    let totalCount = 0;

    for (const level of this.buckets) {
      for (const bucket of level) {
        totalSum += bucket.total;
        totalCount += bucket.count;
      }
    }

    if (totalCount < 10) return false;

    // Check splits
    let leftSum = 0;
    let leftCount = 0;

    for (const level of this.buckets) {
      for (const bucket of level) {
        leftSum += bucket.total;
        leftCount += bucket.count;
        const rightSum = totalSum - leftSum;
        const rightCount = totalCount - leftCount;

        if (leftCount < 5 || rightCount < 5) continue;

        const leftMean = leftSum / leftCount;
        const rightMean = rightSum / rightCount;
        const diff = Math.abs(leftMean - rightMean);

        // Hoeffding bound
        const m = 1 / (1 / leftCount + 1 / rightCount);
        const epsilon = Math.sqrt((1 / (2 * m)) * Math.log(4 / this.delta));

        if (diff > epsilon) {
          // Drift detected, shrink window
          this.shrinkWindow();
          return true;
        }
      }
    }

    return false;
  }

  private shrinkWindow(): void {
    // Remove oldest buckets
    for (let level = this.buckets.length - 1; level >= 0; level--) {
      if (this.buckets[level].length > 0) {
        this.buckets[level].shift();
        break;
      }
    }
  }

  hasChange(): boolean {
    return this.lastChangeDetected;
  }

  getCurrentMean(): number {
    let sum = 0;
    let count = 0;
    for (const level of this.buckets) {
      for (const bucket of level) {
        sum += bucket.total;
        count += bucket.count;
      }
    }
    return count > 0 ? sum / count : 0;
  }

  reset(): void {
    for (const level of this.buckets) {
      level.length = 0;
    }
    this.totalCount = 0;
    this.lastChangeDetected = false;
  }

  serialize(): object {
    return {
      buckets: this.buckets.map((level) =>
        level.map((b) => ({
          total: b.total,
          variance: b.variance,
          count: b.count,
        }))
      ),
      totalCount: this.totalCount,
    };
  }

  deserialize(data: any): void {
    this.buckets = data.buckets.map((level: any[]) =>
      level.map((b: any) => {
        const bucket = new ADWINBucket();
        bucket.total = b.total;
        bucket.variance = b.variance;
        bucket.count = b.count;
        return bucket;
      })
    );
    this.totalCount = data.totalCount;
  }
}

/**
 * Computes sample weight based on prediction error z-score.
 * Downweights outliers to reduce their influence during training.
 */
class OutlierDownweighter {
  private threshold: number;
  private minWeight: number;

  constructor(threshold: number = 3.0, minWeight: number = 0.1) {
    this.threshold = threshold;
    this.minWeight = minWeight;
  }

  /** Compute weight based on error magnitude */
  computeWeight(error: number, errorStd: number): number {
    if (errorStd <= 0) return 1.0;
    const zScore = Math.abs(error) / errorStd;
    if (zScore <= this.threshold) return 1.0;
    return Math.max(this.minWeight, 1 / zScore);
  }
}

/**
 * Tracks running loss and MAE for reporting
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

// ==================== PHASE 9: MAIN API CLASS ====================

/**
 * Resolves configuration with default values
 */
function resolveConfig(
  config: TCNRegressionConfig,
  nFeatures: number,
  nTargets: number,
): ResolvedConfig {
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
    nFeatures,
    nTargets,
  };
}

/**
 * TCNRegression - Main public API class
 *
 * A Temporal Convolutional Network for multivariate regression with:
 * - Online learning (one sample at a time)
 * - Adam optimizer with L2 regularization
 * - Welford z-score normalization
 * - ADWIN drift detection
 * - Uncertainty estimation
 *
 * @example
 * ```typescript
 * const model = new TCNRegression({ maxSequenceLength: 32, hiddenChannels: 16 });
 *
 * // Train online
 * for (const sample of data) {
 *   const result = model.fitOnline({
 *     xCoordinates: [sample.features],
 *     yCoordinates: [sample.targets]
 *   });
 *   console.log(`Loss: ${result.loss}`);
 * }
 *
 * // Predict
 * const prediction = model.predict(3);
 * console.log(prediction.predictions);
 * ```
 */
export class TCNRegression {
  private config: ResolvedConfig | null = null;
  private model: TCNModel | null = null;
  private optimizer: AdamOptimizer | null = null;
  private normalizer: WelfordNormalizer | null = null;
  private inputBuffer: RingBuffer | null = null;
  private residualTracker: ResidualStatsTracker | null = null;
  private adwinDetector: ADWINDetector | null = null;
  private outlierWeighter: OutlierDownweighter | null = null;
  private metricsAccumulator: MetricsAccumulator | null = null;
  private bufferPool: BufferPool;

  // Preallocated training buffers
  private normalizedInput: Float64Array | null = null;
  private normalizedTarget: Float64Array | null = null;
  private inputWindow: Float64Array | null = null;
  private predictions: Float64Array | null = null;
  private dLoss: Float64Array | null = null;
  private forwardContext: ForwardContext | null = null;

  private initialized: boolean = false;
  private sampleCount: number = 0;
  private userConfig: TCNRegressionConfig;

  /**
   * Create a new TCNRegression model
   * @param config Model configuration
   */
  constructor(config: TCNRegressionConfig = {}) {
    this.userConfig = config;
    this.bufferPool = new BufferPool();
  }

  /**
   * Initialize model with inferred dimensions
   */
  private initialize(nFeatures: number, nTargets: number): void {
    if (this.initialized) return;

    this.config = resolveConfig(this.userConfig, nFeatures, nTargets);

    // Create model
    this.model = new TCNModel(
      this.config.nFeatures,
      this.config.nTargets,
      this.config.hiddenChannels,
      this.config.nBlocks,
      this.config.kernelSize,
      this.config.dilationBase,
      this.config.useTwoLayerBlock,
      this.config.activation,
      this.config.useLayerNorm,
      this.config.dropoutRate,
      this.config.maxSequenceLength,
      this.config.maxFutureSteps,
      this.config.useDirectMultiHorizon,
      this.config.seed,
      this.config.weightInitScale,
    );

    // Create optimizer
    this.optimizer = new AdamOptimizer(
      this.config.learningRate,
      this.config.beta1,
      this.config.beta2,
      this.config.epsilon,
      this.config.gradientClipNorm,
      this.config.l2Lambda,
    );

    // Register parameters with optimizer
    for (const [name, param] of this.model.getAllParameters()) {
      this.optimizer.registerParameter(name, param.params.length);
    }

    // Create normalizer
    this.normalizer = new WelfordNormalizer(
      this.config.nFeatures,
      this.config.nTargets,
      this.config.normalizationEpsilon,
      this.config.normalizationWarmup,
    );

    // Create input buffer
    this.inputBuffer = new RingBuffer(
      this.config.maxSequenceLength,
      this.config.nFeatures,
    );

    // Create residual tracker
    this.residualTracker = new ResidualStatsTracker(
      this.config.residualWindowSize,
      this.config.nTargets,
    );

    // Create ADWIN detector if enabled
    if (this.config.adwinEnabled) {
      this.adwinDetector = new ADWINDetector(
        this.config.adwinDelta,
        this.config.adwinMaxBuckets,
      );
    }

    // Create outlier weighter
    this.outlierWeighter = new OutlierDownweighter(
      this.config.outlierThreshold,
      this.config.outlierMinWeight,
    );

    // Create metrics accumulator
    this.metricsAccumulator = new MetricsAccumulator();

    // Allocate training buffers
    this.normalizedInput = new Float64Array(this.config.nFeatures);
    this.normalizedTarget = new Float64Array(this.config.nTargets);
    this.inputWindow = new Float64Array(
      this.config.maxSequenceLength * this.config.nFeatures,
    );

    const outputDim = this.config.useDirectMultiHorizon
      ? this.config.nTargets * this.config.maxFutureSteps
      : this.config.nTargets;
    this.predictions = new Float64Array(outputDim);
    this.dLoss = new Float64Array(outputDim);

    // Create forward context
    this.forwardContext = this.model.createContext(
      this.config.maxSequenceLength,
    );

    this.initialized = true;

    if (this.config.verbose) {
      console.log(
        `TCNRegression initialized: ${nFeatures} features, ${nTargets} targets`,
      );
      console.log(
        `Receptive field: ${this.model.getReceptiveField()} timesteps`,
      );
      console.log(`Total parameters: ${this.model.getParameterCount()}`);
    }
  }

  /**
   * Train the model on a single sample (online learning)
   *
   * @param data Training data with xCoordinates (features) and yCoordinates (targets)
   * @returns FitResult with loss, sample weight, and metrics
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1.0, 2.0, 3.0]],  // Single timestep, 3 features
   *   yCoordinates: [[0.5, 0.8]]         // Single timestep, 2 targets
   * });
   * ```
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    // Validate input
    if (!xCoordinates || xCoordinates.length === 0 || !xCoordinates[0]) {
      throw new Error("xCoordinates must be non-empty");
    }
    if (!yCoordinates || yCoordinates.length === 0 || !yCoordinates[0]) {
      throw new Error("yCoordinates must be non-empty");
    }

    const nFeatures = xCoordinates[0].length;
    const nTargets = yCoordinates[0].length;

    // Initialize on first call
    if (!this.initialized) {
      this.initialize(nFeatures, nTargets);
    }

    // Validate dimensions match
    if (nFeatures !== this.config!.nFeatures) {
      throw new Error(
        `Feature dimension mismatch: expected ${
          this.config!.nFeatures
        }, got ${nFeatures}`,
      );
    }
    if (nTargets !== this.config!.nTargets) {
      throw new Error(
        `Target dimension mismatch: expected ${
          this.config!.nTargets
        }, got ${nTargets}`,
      );
    }

    // Process each timestep in the input
    let lastLoss = 0;
    let lastWeight = 1;
    let driftDetected = false;

    for (let t = 0; t < xCoordinates.length; t++) {
      const x = xCoordinates[t];
      const y = yCoordinates[t];

      // Update normalization statistics
      this.normalizer!.updateInputStats(x);
      this.normalizer!.updateOutputStats(y);

      // Normalize input and push to buffer
      this.normalizer!.normalizeInputsFromArray(this.normalizedInput!, 0, x);
      this.inputBuffer!.push(this.normalizedInput!, 0);

      // Normalize target
      this.normalizer!.normalizeOutputs(this.normalizedTarget!, 0, y);

      this.sampleCount++;

      // Only train if we have enough history
      const seqLen = Math.min(
        this.inputBuffer!.getLength(),
        this.config!.maxSequenceLength,
      );
      if (seqLen < 2) {
        continue;
      }

      // Get input window
      this.inputBuffer!.getWindow(this.inputWindow!, 0, seqLen);

      // Forward pass with context
      this.model!.zeroGradients();
      this.model!.forwardTrain(
        this.predictions!,
        0,
        this.inputWindow!,
        0,
        seqLen,
        this.forwardContext!,
      );

      // Compute loss (MSE)
      const outputDim = this.config!.useDirectMultiHorizon
        ? this.config!.nTargets * this.config!.maxFutureSteps
        : this.config!.nTargets;

      // For single-step prediction, compare first nTargets
      let loss = 0;
      let absError = 0;
      for (let i = 0; i < this.config!.nTargets; i++) {
        const diff = this.predictions![i] - this.normalizedTarget![i];
        loss += diff * diff;
        absError += Math.abs(diff);
      }
      loss /= this.config!.nTargets;
      absError /= this.config!.nTargets;

      // Compute sample weight based on outlier detection
      const residualStats = this.residualTracker!.getStats();
      const avgStd = residualStats.std.reduce((a, b) =>
            a + b, 0) / this.config!.nTargets || 1;
      const sampleWeight = this.outlierWeighter!.computeWeight(
        Math.sqrt(loss),
        avgStd,
      );

      // Compute loss gradient
      LossFunction.mseGradientWeighted(
        this.dLoss!,
        0,
        this.predictions!,
        0,
        this.normalizedTarget!,
        0,
        sampleWeight,
        this.config!.nTargets,
      );

      // Zero remaining gradient elements if direct multi-horizon
      if (this.config!.useDirectMultiHorizon) {
        for (let i = this.config!.nTargets; i < outputDim; i++) {
          this.dLoss![i] = 0;
        }
      }

      // Backward pass
      this.model!.backward(this.dLoss!, 0, this.forwardContext!);

      // Optimizer step
      this.optimizer!.step(this.model!.getAllParameters());

      // Update residual tracker
      this.residualTracker!.addResidual(
        this.predictions!,
        0,
        this.normalizedTarget!,
        0,
      );

      // Update ADWIN detector
      if (this.adwinDetector) {
        driftDetected = this.adwinDetector.update(loss) || driftDetected;
      }

      // Update metrics
      this.metricsAccumulator!.update(loss, absError);

      lastLoss = loss;
      lastWeight = sampleWeight;
    }

    return {
      loss: lastLoss,
      sampleWeight: lastWeight,
      driftDetected,
      metrics: this.metricsAccumulator!.getMetrics(),
    };
  }

  /**
   * Generate predictions for future timesteps
   *
   * @param futureSteps Number of steps to predict (default: 1)
   * @returns PredictionResult with predictions and uncertainty bounds
   *
   * @example
   * ```typescript
   * const result = model.predict(3);
   * console.log(result.predictions);     // [[pred1], [pred2], [pred3]]
   * console.log(result.uncertaintyLower); // Lower bounds
   * console.log(result.uncertaintyUpper); // Upper bounds
   * ```
   */
  predict(futureSteps: number = 1): PredictionResult {
    if (!this.initialized) {
      throw new Error("Model not initialized. Call fitOnline first.");
    }

    if (futureSteps > this.config!.maxFutureSteps) {
      throw new Error(
        `futureSteps (${futureSteps}) exceeds maxFutureSteps (${
          this.config!.maxFutureSteps
        })`,
      );
    }

    const seqLen = Math.min(
      this.inputBuffer!.getLength(),
      this.config!.maxSequenceLength,
    );
    if (seqLen < 1) {
      // Return zeros if no data
      const emptyPred = Array(futureSteps).fill(null).map(() =>
        Array(this.config!.nTargets).fill(0)
      );
      return {
        predictions: emptyPred,
        uncertaintyLower: emptyPred,
        uncertaintyUpper: emptyPred,
        confidence: 0,
      };
    }

    // Get input window
    this.inputBuffer!.getWindow(this.inputWindow!, 0, seqLen);

    // Forward pass
    this.model!.forward(this.predictions!, 0, this.inputWindow!, 0, seqLen);

    // Denormalize predictions
    const outputDim = this.config!.useDirectMultiHorizon
      ? this.config!.nTargets * this.config!.maxFutureSteps
      : this.config!.nTargets;

    const denormalized = new Float64Array(outputDim);

    if (this.config!.useDirectMultiHorizon) {
      // Denormalize each step
      for (let s = 0; s < futureSteps; s++) {
        this.normalizer!.denormalizeOutputs(
          denormalized,
          s * this.config!.nTargets,
          this.predictions!,
          s * this.config!.nTargets,
        );
      }
    } else {
      // Recursive prediction
      const tempInput = new Float64Array(
        this.config!.maxSequenceLength * this.config!.nFeatures,
      );
      TensorOps.copy(
        tempInput,
        0,
        this.inputWindow!,
        0,
        seqLen * this.config!.nFeatures,
      );

      for (let s = 0; s < futureSteps; s++) {
        const currentSeqLen = Math.min(
          seqLen + s,
          this.config!.maxSequenceLength,
        );
        this.model!.forward(this.predictions!, 0, tempInput, 0, currentSeqLen);

        // Store denormalized prediction
        this.normalizer!.denormalizeOutputs(
          denormalized,
          s * this.config!.nTargets,
          this.predictions!,
          0,
        );

        // Roll forward: shift input and append prediction
        if (
          s < futureSteps - 1 && currentSeqLen < this.config!.maxSequenceLength
        ) {
          // Append prediction as next input (simplified: use same as last known)
          const lastIdx = (currentSeqLen - 1) * this.config!.nFeatures;
          const nextIdx = currentSeqLen * this.config!.nFeatures;
          TensorOps.copy(
            tempInput,
            nextIdx,
            tempInput,
            lastIdx,
            this.config!.nFeatures,
          );
        }
      }
    }

    // Build predictions array
    const predictions: number[][] = [];
    for (let s = 0; s < futureSteps; s++) {
      const step: number[] = [];
      for (let t = 0; t < this.config!.nTargets; t++) {
        step.push(denormalized[s * this.config!.nTargets + t]);
      }
      predictions.push(step);
    }

    // Get uncertainty bounds (on normalized scale, then denormalize)
    const bounds = this.residualTracker!.getUncertaintyBounds(
      denormalized,
      0,
      futureSteps,
      this.config!.uncertaintyMultiplier,
    );

    // Compute confidence based on residual tracker count
    const confidence = Math.min(
      1,
      this.residualTracker!.getCount() / this.config!.residualWindowSize,
    );

    return {
      predictions,
      uncertaintyLower: bounds.lower,
      uncertaintyUpper: bounds.upper,
      confidence,
    };
  }

  /**
   * Get model summary including architecture details
   * @returns ModelSummary object
   */
  getModelSummary(): ModelSummary {
    if (!this.initialized) {
      return {
        architecture: "Not initialized",
        layerParams: {},
        totalParams: 0,
        receptiveField: 0,
        memoryBytes: 0,
        nFeatures: 0,
        nTargets: 0,
        sampleCount: 0,
      };
    }

    const layerParams: { [name: string]: number } = {};
    for (const [name, param] of this.model!.getAllParameters()) {
      layerParams[name] = param.params.length;
    }

    const totalParams = this.model!.getParameterCount();
    const receptiveField = this.model!.getReceptiveField();

    // Estimate memory usage (rough)
    const paramBytes = totalParams * 8; // Float64
    const momentBytes = totalParams * 16; // m and v for Adam
    const activationBytes = this.config!.maxSequenceLength *
      this.config!.hiddenChannels * 8 * this.config!.nBlocks;
    const bufferBytes = this.inputWindow!.length * 8 +
      this.predictions!.length * 8;
    const memoryBytes = paramBytes + momentBytes + activationBytes +
      bufferBytes;

    const architecture = [
      `TCN Backbone: ${this.config!.nBlocks} blocks`,
      `  Channels: ${this.config!.nFeatures} -> ${this.config!.hiddenChannels}`,
      `  Kernel size: ${this.config!.kernelSize}`,
      `  Dilation base: ${this.config!.dilationBase}`,
      `  Two-layer blocks: ${this.config!.useTwoLayerBlock}`,
      `  Activation: ${this.config!.activation}`,
      `  Layer norm: ${this.config!.useLayerNorm}`,
      `Output Head: ${
        this.config!.useDirectMultiHorizon ? "Direct" : "Recursive"
      }`,
      `  Hidden -> ${this.config!.nTargets}${
        this.config!.useDirectMultiHorizon
          ? ` x ${this.config!.maxFutureSteps}`
          : ""
      }`,
    ].join("\n");

    return {
      architecture,
      layerParams,
      totalParams,
      receptiveField,
      memoryBytes,
      nFeatures: this.config!.nFeatures,
      nTargets: this.config!.nTargets,
      sampleCount: this.sampleCount,
    };
  }

  /**
   * Get all model weights
   * @returns WeightInfo object with all parameters
   */
  getWeights(): WeightInfo {
    if (!this.initialized) {
      return { parameters: {} };
    }

    const parameters: WeightInfo["parameters"] = {};
    const allParams = this.model!.getAllParameters();

    for (const [name, param] of allParams) {
      const parts = name.split("_");
      const layerName = parts.slice(0, -1).join("_");
      const paramType = parts[parts.length - 1];

      if (!parameters[layerName]) {
        parameters[layerName] = {};
      }

      if (paramType === "weight") {
        parameters[layerName].weights = {
          data: Array.from(param.params),
          shape: [param.params.length],
        };
      } else if (paramType === "bias") {
        parameters[layerName].bias = {
          data: Array.from(param.params),
          shape: [param.params.length],
        };
      }
    }

    return { parameters };
  }

  /**
   * Get normalization statistics
   * @returns NormalizationStats object
   */
  getNormalizationStats(): NormalizationStats {
    if (!this.initialized || !this.normalizer) {
      return {
        inputMeans: [],
        inputStds: [],
        outputMeans: [],
        outputStds: [],
        sampleCount: 0,
        isWarmedUp: false,
      };
    }

    return this.normalizer.getStats();
  }

  /**
   * Reset model to initial state
   * Re-initializes all parameters, optimizer state, and buffers
   */
  reset(): void {
    if (!this.initialized) return;

    // Re-initialize model
    this.model!.reinitialize(this.config!.seed, this.config!.weightInitScale);

    // Reset optimizer
    this.optimizer!.reset();

    // Re-register parameters
    for (const [name, param] of this.model!.getAllParameters()) {
      const state = this.optimizer!.getState(name);
      if (state) state.reset();
    }

    // Reset normalizer
    this.normalizer!.reset();

    // Reset buffers
    this.inputBuffer!.clear();
    this.residualTracker!.reset();
    if (this.adwinDetector) this.adwinDetector.reset();
    this.metricsAccumulator!.reset();

    this.sampleCount = 0;
  }

  /**
   * Serialize model state to JSON string
   * @returns JSON string containing all model state
   */
  save(): string {
    if (!this.initialized) {
      throw new Error("Model not initialized. Nothing to save.");
    }

    const state = {
      version: "1.0.0",
      config: this.userConfig,
      resolvedConfig: this.config,
      sampleCount: this.sampleCount,
      parameters: {} as { [key: string]: number[] },
      optimizer: this.optimizer!.serialize(),
      normalizer: this.normalizer!.serialize(),
      inputBuffer: this.inputBuffer!.serialize(),
      residualTracker: this.residualTracker!.serialize(),
      adwinDetector: this.adwinDetector ? this.adwinDetector.serialize() : null,
    };

    // Save parameters
    for (const [name, param] of this.model!.getAllParameters()) {
      state.parameters[name] = Array.from(param.params);
    }

    return JSON.stringify(state);
  }

  /**
   * Load model state from JSON string
   * @param json JSON string from save()
   */
  load(json: string): void {
    const state = JSON.parse(json);

    if (!state.version || !state.resolvedConfig) {
      throw new Error("Invalid save format");
    }

    // Re-initialize with saved config
    this.userConfig = state.config;
    this.config = state.resolvedConfig;
    this.sampleCount = state.sampleCount;

    // Initialize model structure
    this.initialize(this.config!.nFeatures, this.config!.nTargets);

    // Restore parameters
    for (const [name, param] of this.model!.getAllParameters()) {
      if (state.parameters[name]) {
        for (let i = 0; i < param.params.length; i++) {
          param.params[i] = state.parameters[name][i];
        }
      }
    }

    // Restore optimizer state
    this.optimizer!.deserialize(state.optimizer);

    // Restore normalizer
    this.normalizer!.deserialize(state.normalizer);

    // Restore input buffer
    this.inputBuffer!.deserialize(state.inputBuffer);

    // Restore residual tracker
    this.residualTracker!.deserialize(state.residualTracker);

    // Restore ADWIN detector
    if (this.adwinDetector && state.adwinDetector) {
      this.adwinDetector.deserialize(state.adwinDetector);
    }
  }
}

// Default export
export default TCNRegression;
