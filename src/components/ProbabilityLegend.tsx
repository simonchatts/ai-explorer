import { probabilityColor } from "../utils/probabilityColor";

const LEGEND_VALUES = [0, 0.25, 0.5, 0.75, 1];

export function ProbabilityLegend({ visible }: { visible: boolean }) {
  if (!visible) return null;

  return (
    <section className="panel legend-panel" aria-labelledby="legend-title">
      <div className="panel-heading">
        <h2 id="legend-title">Probability</h2>
      </div>
      <div className="legend-swatches">
        {LEGEND_VALUES.map((value) => (
          <div className="legend-item" key={value}>
            <span style={{ backgroundColor: probabilityColor(value) }} aria-hidden="true" />
            <strong>{Math.round(value * 100)}%</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
