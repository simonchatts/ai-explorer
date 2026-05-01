import type { TokenCandidate } from "./types";

export function sampleVisibleToken(
  candidates: TokenCandidate[],
  random: () => number = Math.random,
): TokenCandidate | null {
  if (candidates.length === 0) return null;

  const total = candidates.reduce((sum, candidate) => sum + Math.max(0, candidate.probability), 0);
  if (total <= 0) {
    return candidates[Math.floor(random() * candidates.length)] ?? candidates[0];
  }

  let threshold = random() * total;
  for (const candidate of candidates) {
    threshold -= Math.max(0, candidate.probability);
    if (threshold <= 0) return candidate;
  }

  return candidates[candidates.length - 1];
}
