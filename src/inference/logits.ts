import type { TokenCandidate } from "./types";

export function probabilityForToken(rowLogits: ArrayLike<number>, tokenId: number): number {
  if (tokenId < 0 || tokenId >= rowLogits.length) {
    return 0;
  }

  let max = -Infinity;
  for (let i = 0; i < rowLogits.length; i += 1) {
    const value = rowLogits[i];
    if (value > max) max = value;
  }

  let sum = 0;
  let target = 0;
  for (let i = 0; i < rowLogits.length; i += 1) {
    const value = Math.exp(rowLogits[i] - max);
    sum += value;
    if (i === tokenId) target = value;
  }

  return sum === 0 ? 0 : target / sum;
}

export function topTokensFromLogits(
  rowLogits: ArrayLike<number>,
  n: number,
  decodeToken: (tokenId: number) => string,
): TokenCandidate[] {
  const limit = Math.max(0, Math.min(n, rowLogits.length));
  if (limit === 0) return [];

  let max = -Infinity;
  for (let i = 0; i < rowLogits.length; i += 1) {
    const value = rowLogits[i];
    if (value > max) max = value;
  }

  let denominator = 0;
  const best: Array<{ tokenId: number; logit: number }> = [];

  for (let tokenId = 0; tokenId < rowLogits.length; tokenId += 1) {
    const logit = rowLogits[tokenId];
    denominator += Math.exp(logit - max);

    if (best.length < limit) {
      best.push({ tokenId, logit });
      best.sort((a, b) => a.logit - b.logit);
      continue;
    }

    if (logit > best[0].logit) {
      best[0] = { tokenId, logit };
      best.sort((a, b) => a.logit - b.logit);
    }
  }

  return best
    .sort((a, b) => b.logit - a.logit)
    .map(({ tokenId, logit }) => ({
      tokenId,
      text: decodeToken(tokenId),
      probability: denominator === 0 ? 0 : Math.exp(logit - max) / denominator,
    }));
}
