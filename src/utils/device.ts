export interface DeviceSignalInput {
  userAgent: string;
  width: number;
  deviceMemory?: number;
}

export interface DeviceLoadDecision {
  shouldAutoLoad: boolean;
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

export function getWebGpuStatus(): "available" | "unavailable" {
  return "gpu" in navigator ? "available" : "unavailable";
}
