export function clampProbability(probability: number): number {
  if (!Number.isFinite(probability)) return 0;
  return Math.max(0, Math.min(1, probability));
}

export function probabilityToPercent(probability: number): string {
  return `${(clampProbability(probability) * 100).toFixed(2)}%`;
}

export function probabilityColor(probability: number): string {
  const p = clampProbability(probability);
  const hue = 24 + p * 168;
  const saturation = 82 - p * 18;
  const lightness = 88 - p * 30;
  return `hsl(${hue.toFixed(1)} ${saturation.toFixed(1)}% ${lightness.toFixed(1)}%)`;
}
