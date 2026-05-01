import { AlertTriangle, Cpu, Download, RotateCcw, X } from "lucide-react";
import type { LoadStatus } from "../state/explorerStore";

interface ModelLoadGateProps {
  status: LoadStatus;
  progress: number | null;
  message: string | null;
  error: string | null;
  consentReason: string;
  onContinue: () => void;
  onCancel: () => void;
  onRetry: () => void;
}

export function ModelLoadGate({
  status,
  progress,
  message,
  error,
  consentReason,
  onContinue,
  onCancel,
  onRetry,
}: ModelLoadGateProps) {
  if (status === "ready" || status === "idle") return null;

  const progressValue = progress == null ? undefined : Math.max(0, Math.min(100, progress));

  return (
    <div className="load-backdrop" role="presentation">
      <section className="load-panel" role="dialog" aria-modal="true" aria-labelledby="load-title">
        {status === "needs-consent" ? (
          <>
            <div className="load-icon">
              <Cpu aria-hidden="true" />
            </div>
            <h2 id="load-title">Load the local model?</h2>
            <p>
              This app downloads a browser-ready Qwen model and runs it with WebGPU on your device.
              {consentReason ? ` ${consentReason}` : ""}
            </p>
            <div className="load-actions">
              <button type="button" className="primary-button" onClick={onContinue}>
                <Download size={18} aria-hidden="true" />
                Continue and download
              </button>
              <button type="button" className="ghost-button" onClick={onCancel}>
                <X size={18} aria-hidden="true" />
                Cancel
              </button>
            </div>
          </>
        ) : null}

        {status === "loading" ? (
          <>
            <div className="load-icon">
              <Download aria-hidden="true" />
            </div>
            <h2 id="load-title">Loading model</h2>
            <p>{message ?? "Preparing tokenizer and model weights."}</p>
            <progress value={progressValue} max={100} aria-label="Model loading progress" />
            <span className="progress-label">
              {progressValue == null ? "Working..." : `${progressValue.toFixed(0)}%`}
            </span>
          </>
        ) : null}

        {status === "error" ? (
          <>
            <div className="load-icon error">
              <AlertTriangle aria-hidden="true" />
            </div>
            <h2 id="load-title">Model could not load</h2>
            <p>{error ?? "Something went wrong while preparing local inference."}</p>
            <div className="load-actions">
              <button type="button" className="primary-button" onClick={onRetry}>
                <RotateCcw size={18} aria-hidden="true" />
                Retry
              </button>
            </div>
          </>
        ) : null}
      </section>
    </div>
  );
}
