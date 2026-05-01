interface StatusBarProps {
  modelId: string;
  webGpuStatus: "available" | "unavailable";
  tokenCount: number;
  completionTokenCount: number;
  activity: string;
}

export function StatusBar({
  modelId,
  webGpuStatus,
  tokenCount,
  completionTokenCount,
  activity,
}: StatusBarProps) {
  return (
    <section className="panel status-panel" aria-labelledby="status-title">
      <div className="panel-heading">
        <h2 id="status-title">Status</h2>
      </div>
      <dl>
        <div>
          <dt>Model</dt>
          <dd>{modelId}</dd>
        </div>
        <div>
          <dt>WebGPU</dt>
          <dd>{webGpuStatus}</dd>
        </div>
        <div>
          <dt>Tokens</dt>
          <dd>
            {tokenCount} total / {completionTokenCount} generated
          </dd>
        </div>
        <div>
          <dt>Activity</dt>
          <dd>{activity}</dd>
        </div>
      </dl>
    </section>
  );
}
