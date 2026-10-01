import { describe, expect, it, vi } from "vitest";
import { InferenceCache } from "./inferenceCache";
import type { ForwardInputs, ForwardOutputs } from "./inferenceCache";
import { probabilityForToken, topTokensFromLogits } from "./logits";

function tensor(ids: number[] = []) {
  return { ids, location: "gpu-buffer", dispose: vi.fn() };
}
type FakeTensor = ReturnType<typeof tensor>;

// A causal model whose logits depend on the entire prefix, not just the last token.
function logitsFor(ids: number[]): Float32Array {
  const sum = ids.reduce((total, id, index) => total + id * (index + 1), 0);
  return Float32Array.from([Math.sin(sum), Math.cos(sum), sum / 10, -sum / 10]);
}

function fixture() {
  const outputs: ForwardOutputs<FakeTensor>[] = [];
  const forward = vi.fn(async (inputs: ForwardInputs<FakeTensor>): Promise<ForwardOutputs<FakeTensor>> => {
    const prefix = inputs.pastKeyValues?.key.ids ?? [];
    expect(inputs.positionOffset).toBe(prefix.length);
    expect(inputs.attentionLength).toBe(prefix.length + inputs.tokenIds.length);
    const ids = [...prefix, ...inputs.tokenIds];
    const rows = inputs.tokenIds.map((_, index) => Array.from(logitsFor(ids.slice(0, prefix.length + index + 1))));
    const output = {
      logits: { ...tensor(), dims: [1, rows.length, 4], data: Float32Array.from(rows.flat()) },
      pastKeyValues: { key: tensor(ids), value: tensor(ids) },
    };
    outputs.push(output);
    return output;
  });
  return { cache: new InferenceCache(forward), forward, outputs };
}

function expectedProbabilities(ids: number[]): number[] {
  return ids.map((id, index) => index === 0 ? 0.5 : probabilityForToken(logitsFor(ids.slice(0, index)), id));
}

describe("InferenceCache", () => {
  it("shares a forward pass for concurrent predictions and probabilities, and reuses results for more candidates", async () => {
    const { cache, forward } = fixture();
    const ids = [0, 1, 2];
    const [predictions, scores] = await Promise.all([cache.infer(ids, true), cache.infer(ids, true)]);
    expect(forward).toHaveBeenCalledTimes(1);
    expect(predictions!.nextLogits).toEqual(logitsFor(ids));
    expect(scores!.tokenProbabilities).toEqual(expectedProbabilities(ids));
    await cache.infer(ids, false);
    expect(forward).toHaveBeenCalledTimes(1);
    scores!.tokenProbabilities![0] = 123;
    expect((await cache.infer(ids, true))!.tokenProbabilities![0]).toBe(0.5);
  });

  it("processes only appended tokens while matching uncached candidates and every token probability", async () => {
    const { cache, forward, outputs } = fixture();
    await cache.infer([0, 1], true);
    for (const ids of [[0, 1, 2], [0, 1, 2, 3, 1]]) {
      const result = (await cache.infer(ids, true))!;
      expect(result.nextLogits).toEqual(logitsFor(ids));
      expect(result.tokenProbabilities).toEqual(expectedProbabilities(ids));
      expect(topTokensFromLogits(result.nextLogits, 3, String)).toEqual(topTokensFromLogits(logitsFor(ids), 3, String));
    }
    expect(forward.mock.calls.map(([input]) => input.tokenIds)).toEqual([[0, 1], [2], [3, 1]]);
    expect(outputs[0].pastKeyValues.key.dispose).toHaveBeenCalledOnce();
    expect(outputs[1].pastKeyValues.key.dispose).toHaveBeenCalledOnce();
    expect(outputs[2].pastKeyValues.key.dispose).not.toHaveBeenCalled();
    for (const output of outputs) expect(output.logits.dispose).toHaveBeenCalledOnce();
  });

  it("rebuilds on deletion, replacement, and branching, then resumes incremental inference", async () => {
    const { cache, forward, outputs } = fixture();
    for (const ids of [[0, 1, 2], [0, 1], [0, 3], [0, 3, 2]]) {
      const result = (await cache.infer(ids, true))!;
      expect(result.nextLogits).toEqual(logitsFor(ids));
      expect(result.tokenProbabilities).toEqual(expectedProbabilities(ids));
    }
    expect(forward.mock.calls.map(([input]) => input.tokenIds)).toEqual([[0, 1, 2], [0, 1], [0, 3], [2]]);
    for (const output of outputs.slice(0, -1)) expect(output.pastKeyValues.key.dispose).toHaveBeenCalledOnce();
  });

  it("skips prompt scoring while hidden, rebuilds once on reveal, and retains scores across later hidden steps", async () => {
    const { cache, forward } = fixture();
    expect((await cache.infer([0, 1], false))!.tokenProbabilities).toBeNull();
    expect((await cache.infer([0, 1, 2], false))!.tokenProbabilities).toBeNull();
    const revealed = await cache.infer([0, 1, 2], true);
    expect(revealed!.tokenProbabilities).toEqual(expectedProbabilities([0, 1, 2]));
    await cache.infer([0, 1, 2, 3], false);
    expect((await cache.infer([0, 1, 2, 3], true))!.tokenProbabilities).toEqual(expectedProbabilities([0, 1, 2, 3]));
    expect(forward.mock.calls.map(([input]) => input.tokenIds)).toEqual([[0, 1], [2], [0, 1, 2], [3]]);
  });

  it("serializes different sequences, so GPU state cannot be reused before the preceding forward finishes", async () => {
    const { cache, forward } = fixture();
    const results = await Promise.all([cache.infer([0, 1], true), cache.infer([0, 1, 2], true), cache.infer([3], true)]);
    expect(results.map((result) => result!.nextLogits)).toEqual([logitsFor([0, 1]), logitsFor([0, 1, 2]), logitsFor([3])]);
    expect(forward.mock.calls.map(([input]) => input.tokenIds)).toEqual([[0, 1], [2], [3]]);
  });

  it("clears the cache on an empty prompt and preserves the first-token probability convention", async () => {
    const { cache, forward, outputs } = fixture();
    expect((await cache.infer([1], true))!.tokenProbabilities).toEqual([0.5]);
    expect(await cache.infer([], true)).toBeNull();
    expect(outputs[0].pastKeyValues.key.dispose).toHaveBeenCalledOnce();
    expect((await cache.infer([1], true))!.tokenProbabilities).toEqual([0.5]);
    expect(forward).toHaveBeenCalledTimes(2);
  });

  it("releases stale GPU state after a failed forward and allows the next request to rebuild", async () => {
    const { cache, forward, outputs } = fixture();
    await cache.infer([0, 1], true);
    forward.mockRejectedValueOnce(new Error("GPU failure"));
    await expect(cache.infer([0, 1, 2], true)).rejects.toThrow("GPU failure");
    expect(outputs[0].pastKeyValues.key.dispose).toHaveBeenCalledOnce();
    expect((await cache.infer([0, 1, 2], true))!.nextLogits).toEqual(logitsFor([0, 1, 2]));
    expect(forward.mock.calls.at(-1)![0].pastKeyValues).toBeNull();
  });

  it("releases newly returned GPU tensors if the model returns invalid logits", async () => {
    const { cache, forward } = fixture();
    const badOutput = {
      logits: { ...tensor(), dims: [1, 99, 4], data: new Float32Array(4) },
      pastKeyValues: { key: tensor() },
    };
    forward.mockResolvedValueOnce(badOutput as ForwardOutputs<FakeTensor>);
    await expect(cache.infer([1], true)).rejects.toThrow("Unexpected logits shape");
    expect(badOutput.logits.dispose).toHaveBeenCalledOnce();
    expect(badOutput.pastKeyValues.key.dispose).toHaveBeenCalledOnce();
    expect((await cache.infer([1], true))!.nextLogits).toEqual(logitsFor([1]));
  });
});
