import { describe, expect, it } from "vitest";
import { probabilityForToken, topTokensFromLogits } from "./logits";

describe("logit utilities", () => {
  it("computes a stable softmax probability for one token", () => {
    const probability = probabilityForToken([1000, 1001, 1002], 2);
    expect(probability).toBeCloseTo(0.6652, 4);
  });

  it("selects the top tokens without sorting the whole vocabulary", () => {
    const candidates = topTokensFromLogits([0, 6, 2, 5, 1], 2, (tokenId) => `tok-${tokenId}`);

    expect(candidates.map((candidate) => candidate.tokenId)).toEqual([1, 3]);
    expect(candidates[0].text).toBe("tok-1");
    expect(candidates[0].probability).toBeGreaterThan(candidates[1].probability);
  });
});
