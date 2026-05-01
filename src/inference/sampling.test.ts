import { describe, expect, it } from "vitest";
import { sampleVisibleToken } from "./sampling";
import type { TokenCandidate } from "./types";

const candidates: TokenCandidate[] = [
  { tokenId: 1, text: "a", probability: 0.2 },
  { tokenId: 2, text: "b", probability: 0.3 },
  { tokenId: 3, text: "c", probability: 0.5 },
];

describe("sampleVisibleToken", () => {
  it("samples only from visible candidates using their visible weights", () => {
    expect(sampleVisibleToken(candidates, () => 0.0)?.tokenId).toBe(1);
    expect(sampleVisibleToken(candidates, () => 0.21)?.tokenId).toBe(2);
    expect(sampleVisibleToken(candidates, () => 0.99)?.tokenId).toBe(3);
  });

  it("falls back to uniform sampling when all probabilities are zero", () => {
    const zeroes = candidates.map((candidate) => ({ ...candidate, probability: 0 }));
    expect(sampleVisibleToken(zeroes, () => 0.7)?.tokenId).toBe(3);
  });
});
