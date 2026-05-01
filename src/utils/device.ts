export interface DeviceSignalInput {
  userAgent: string;
  width: number;
  deviceMemory?: number;
}

export interface WebGpuSignalInput {
  hasGpu: boolean;
  requestAdapter?: () => Promise<unknown | null>;
}

export interface DeviceLoadDecision {
  shouldAutoLoad: boolean;
  reason: string;
}

export interface WebGpuSupportDecision {
  isSupported: boolean;
  reason: string;
}

const MOBILE_PATTERN = /Mobi|Android|iPhone|iPad|iPod/i;

export function getDeviceLoadDecision(input: DeviceSignalInput): DeviceLoadDecision {
  if (MOBILE_PATTERN.test(input.userAgent)) {
    return { shouldAutoLoad: false, reason: "This looks like a mobile or tablet browser." };
  }

  if (input.width < 900) {
    return { shouldAutoLoad: false, reason: "The viewport is narrow for a local model workbench." };
  }

  if (typeof input.deviceMemory === "number" && input.deviceMemory < 4) {
    return { shouldAutoLoad: false, reason: "This device reports less than 4 GB of memory." };
  }

  return { shouldAutoLoad: true, reason: "Desktop-like browser detected." };
}

export function currentDeviceLoadDecision(): DeviceLoadDecision {
  const memory = "deviceMemory" in navigator ? Number(navigator.deviceMemory) : undefined;
  return getDeviceLoadDecision({
    userAgent: navigator.userAgent,
    width: window.innerWidth,
    deviceMemory: Number.isFinite(memory) ? memory : undefined,
  });
}

export async function getWebGpuSupport(
  input: WebGpuSignalInput,
): Promise<WebGpuSupportDecision> {
  if (!input.hasGpu) {
    return {
      isSupported: false,
      reason:
        "WebGPU is not available in this browser. Try desktop Chrome or Edge with WebGPU enabled.",
    };
  }

  if (!input.requestAdapter) {
    return {
      isSupported: false,
      reason:
        "WebGPU is present, but this browser does not expose the adapter request API.",
    };
  }

  try {
    const adapter = await input.requestAdapter();
    if (!adapter) {
      return {
        isSupported: false,
        reason:
          "WebGPU is present, but no compatible GPU adapter is available in this browser.",
      };
    }
  } catch (error) {
    return {
      isSupported: false,
      reason: `WebGPU adapter initialization failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    };
  }

  return { isSupported: true, reason: "WebGPU is available." };
}

export function currentWebGpuSupport(): Promise<WebGpuSupportDecision> {
  const gpu = (
    navigator as {
      gpu?: {
        requestAdapter?: () => Promise<unknown | null>;
      };
    }
  ).gpu;

  return getWebGpuSupport({
    hasGpu: Boolean(gpu),
    requestAdapter: gpu?.requestAdapter?.bind(gpu),
  });
}
