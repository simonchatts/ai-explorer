import { describe, expect, it } from "vitest";
import { clampProbability, probabilityColor, probabilityToPercent } from "./probabilityColor";

describe("probability color helpers", () => {
  it("clamps invalid probability values", () => {
    expect(clampProbability(-1)).toBe(0);
    expect(clampProbability(2)).toBe(1);
    expect(clampProbability(Number.NaN)).toBe(0);
  });

  it("formats percentages with numeric labels", () => {
    expect(probabilityToPercent(0.12345)).toBe("12.35%");
  });

  it("maps probabilities to hsl colors", () => {
    expect(probabilityColor(0)).toContain("hsl(");
    expect(probabilityColor(0)).not.toBe(probabilityColor(1));
  });
});
