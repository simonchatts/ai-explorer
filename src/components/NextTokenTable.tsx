import type { TokenCandidate } from "../inference/types";
import { probabilityToPercent } from "../utils/probabilityColor";

interface NextTokenTableProps {
  candidates: TokenCandidate[];
  selectedTokenId: number | null;
  showTokenIds: boolean;
  disabled: boolean;
  onAppend: (candidate: TokenCandidate) => void;
}

function labelForCandidate(candidate: TokenCandidate, showTokenIds: boolean): string {
  if (showTokenIds) return String(candidate.tokenId);
  return candidate.text.replaceAll("\n", "\\n").replaceAll("\t", "\\t") || " ";
}

export function NextTokenTable({
  candidates,
  selectedTokenId,
  showTokenIds,
  disabled,
  onAppend,
}: NextTokenTableProps) {
  const maxProbability = Math.max(...candidates.map((candidate) => candidate.probability), 0);

  return (
    <section className="panel token-table-panel" aria-labelledby="next-token-title">
      <div className="panel-heading">
        <h2 id="next-token-title">Next Tokens</h2>
        <span>{candidates.length} candidates</span>
      </div>
      <div className="candidate-table" role="table" aria-label="Next token candidates">
        <div className="candidate-row header" role="row">
          <span role="columnheader">{showTokenIds ? "Token id" : "Token"}</span>
          <span role="columnheader">Probability</span>
        </div>
        {candidates.map((candidate) => {
          const width = maxProbability > 0 ? `${(candidate.probability / maxProbability) * 100}%` : "0%";
          const selected = selectedTokenId === candidate.tokenId;

          return (
            <button
              type="button"
              className={selected ? "candidate-row selected" : "candidate-row"}
              role="row"
              key={candidate.tokenId}
              disabled={disabled}
              onClick={() => onAppend(candidate)}
            >
              <span className="candidate-token" role="cell">
                {labelForCandidate(candidate, showTokenIds)}
              </span>
              <span className="candidate-probability" role="cell">
                <span className="probability-track" aria-hidden="true">
                  <span style={{ width }} />
                </span>
                <span>{probabilityToPercent(candidate.probability)}</span>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
