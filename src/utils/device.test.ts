import { describe, expect, it } from "vitest";
import { getDeviceLoadDecision } from "./device";

describe("getDeviceLoadDecision", () => {
  it("auto-loads on desktop-like devices", () => {
    expect(
      getDeviceLoadDecision({
        userAgent: "Mozilla/5.0 Macintosh",
        width: 1440,
        deviceMemory: 8,
      }).shouldAutoLoad,
    ).toBe(true);
  });

  it("requires consent on mobile user agents", () => {
    expect(
      getDeviceLoadDecision({
        userAgent: "Mozilla/5.0 iPhone",
        width: 1200,
        deviceMemory: 8,
      }).shouldAutoLoad,
    ).toBe(false);
  });

  it("requires consent on narrow or low-memory devices", () => {
    expect(
      getDeviceLoadDecision({
        userAgent: "Mozilla/5.0 Macintosh",
        width: 700,
        deviceMemory: 8,
      }).shouldAutoLoad,
    ).toBe(false);
    expect(
      getDeviceLoadDecision({
        userAgent: "Mozilla/5.0 Macintosh",
        width: 1400,
        deviceMemory: 2,
      }).shouldAutoLoad,
    ).toBe(false);
  });
});
