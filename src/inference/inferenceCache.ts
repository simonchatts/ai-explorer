import { probabilityForToken } from "./logits";

export interface CacheTensor {
  readonly location: string;
  dispose(): void;
}

export interface ForwardInputs<T extends CacheTensor> {
  tokenIds: number[];
  positionOffset: number;
  attentionLength: number;
  pastKeyValues: Record<string, T> | null;
}

export interface ForwardOutputs<T extends CacheTensor> {
  logits: CacheTensor & { dims: number[]; data: Float32Array };
  pastKeyValues: Record<string, T>;
}

interface CachedPrediction {
  nextLogits: Float32Array;
  tokenProbabilities: number[] | null;
}

/** Owns one sequence's KV tensors. All inference requests run in order. */
export class InferenceCache<T extends CacheTensor> {
  private tokenIds: number[] = [];
  private pastKeyValues: Record<string, T> | null = null;
  private prediction: CachedPrediction | null = null;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(private readonly forward: (inputs: ForwardInputs<T>) => Promise<ForwardOutputs<T>>) {}

  infer(tokenIds: number[], includeProbabilities: boolean): Promise<CachedPrediction | null> {
    const ids = [...tokenIds];
    const result = this.queue.then(() => this.run(ids, includeProbabilities));
    // A failed request must not block subsequent requests or overlap cache mutations.
    this.queue = result.catch(() => undefined);
    return result;
  }

  private releasePast(values: Record<string, T> | null, retained: Record<string, T> = {}): void {
    const keep = new Set(Object.values(retained));
    for (const tensor of new Set(Object.values(values ?? {}))) {
      if (!keep.has(tensor) && tensor.location === "gpu-buffer") tensor.dispose();
    }
  }

  private reset(): void {
    this.releasePast(this.pastKeyValues);
    this.tokenIds = [];
    this.pastKeyValues = null;
    this.prediction = null;
  }

  private async run(tokenIds: number[], includeProbabilities: boolean): Promise<CachedPrediction | null> {
    if (tokenIds.length === 0) {
      this.reset();
      return null;
    }

    const samePrefix = this.tokenIds.length <= tokenIds.length &&
      this.tokenIds.every((id, index) => id === tokenIds[index]);
    const needsScores = includeProbabilities && !this.prediction?.tokenProbabilities;
    if (samePrefix && tokenIds.length === this.tokenIds.length && this.prediction && !needsScores) {
      return this.snapshot();
    }

    // A late probability toggle needs a prefill to score the earlier prompt tokens.
    // Edits and backtracking also rebuild instead of reusing an incompatible cache.
    const incremental = samePrefix && !!this.prediction && !!this.pastKeyValues &&
      Object.keys(this.pastKeyValues).length > 0 && !needsScores;
    if (!incremental) this.reset();
    const offset = this.tokenIds.length;
    let outputs: ForwardOutputs<T> | null = null;
    let committed = false;
    try {
      outputs = await this.forward({
        tokenIds: tokenIds.slice(offset),
        positionOffset: offset,
        attentionLength: tokenIds.length,
        pastKeyValues: this.pastKeyValues,
      });
      const [batch, length, vocabSize] = outputs.logits.dims;
      if (outputs.logits.dims.length !== 3 || batch !== 1 || length !== tokenIds.length - offset) {
        throw new Error(`Unexpected logits shape [${outputs.logits.dims.join(", ")}].`);
      }
      const data = outputs.logits.data;
      const row = (index: number): Float32Array => data.subarray(index * vocabSize, (index + 1) * vocabSize);
      let probabilities: number[] | null = null;
      if (includeProbabilities || this.prediction?.tokenProbabilities) {
        probabilities = incremental ? [...this.prediction!.tokenProbabilities!] : [0.5];
        for (let index = Math.max(1, offset); index < tokenIds.length; index += 1) {
          const previousRow = index === offset
            ? this.prediction!.nextLogits
            : row(index - offset - 1);
          probabilities.push(probabilityForToken(previousRow, tokenIds[index]));
        }
      }
      // Keep only one vocabulary row, not the entire prefill's logits buffer.
      const nextLogits = row(length - 1).slice();
      this.releasePast(this.pastKeyValues, outputs.pastKeyValues);
      this.pastKeyValues = outputs.pastKeyValues;
      this.tokenIds = tokenIds;
      this.prediction = { nextLogits, tokenProbabilities: probabilities };
      committed = true;
      return this.snapshot();
    } catch (error) {
      this.reset();
      throw error;
    } finally {
      if (outputs) {
        if (outputs.logits.location === "gpu-buffer") outputs.logits.dispose();
        if (!committed) this.releasePast(outputs.pastKeyValues);
      }
    }
  }

  private snapshot(): CachedPrediction {
    return {
      nextLogits: this.prediction!.nextLogits,
      tokenProbabilities: this.prediction!.tokenProbabilities?.slice() ?? null,
    };
  }
}
