import { describe, expect, it } from "vitest";
import { getDeviceLoadDecision, getWebGpuSupport } from "./device";

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

describe("getWebGpuSupport", () => {
  it("rejects browsers without WebGPU", () => {
    return expect(
      getWebGpuSupport({
        hasGpu: false,
      }),
    ).resolves.toMatchObject({ isSupported: false });
  });

  it("rejects WebGPU without adapter access", async () => {
    const support = await getWebGpuSupport({
      hasGpu: true,
    });

    expect(support.isSupported).toBe(false);
    expect(support.reason).toContain("adapter request API");
  });

  it("rejects WebGPU when no adapter is available", async () => {
    const support = await getWebGpuSupport({
      hasGpu: true,
      requestAdapter: () => Promise.resolve(null),
    });

    expect(support.isSupported).toBe(false);
    expect(support.reason).toContain("no compatible GPU adapter");
  });

  it("accepts browsers with an available WebGPU adapter", () => {
    return expect(
      getWebGpuSupport({
        hasGpu: true,
        requestAdapter: () => Promise.resolve({ name: "test adapter" }),
      }),
    ).resolves.toMatchObject({ isSupported: true });
  });
});
