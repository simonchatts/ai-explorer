import { probabilityColor, probabilityToPercent } from "../utils/probabilityColor";

interface CompletionViewProps {
  decodedText: string;
  tokenIds: number[];
  tokenTexts: string[];
  tokenProbabilities: number[];
  baseTokenCount: number;
  showTokenIds: boolean;
  showProbabilities: boolean;
}

function visibleTokenText(text: string): string {
  if (text === "\n") return "\\n";
  if (text === "\t") return "\\t";
  return text.replaceAll("\n", "\\n").replaceAll("\t", "\\t") || " ";
}

export function CompletionView({
  decodedText,
  tokenIds,
  tokenTexts,
  tokenProbabilities,
  baseTokenCount,
  showTokenIds,
  showProbabilities,
}: CompletionViewProps) {
  const idOutput = tokenIds.join(" ");

  return (
    <section className="panel completion-panel" aria-labelledby="completion-title">
      <div className="panel-heading">
        <h2 id="completion-title">Current Prompt / Completion</h2>
        <span>{tokenIds.length} tokens</span>
      </div>

      {showProbabilities ? (
        <div className="token-chip-wrap" aria-label="Tokens with probabilities">
          {tokenIds.map((tokenId, index) => {
            const probability = tokenProbabilities[index] ?? 0;
            return (
              <span
                className={index >= baseTokenCount ? "token-chip generated" : "token-chip"}
                style={{ backgroundColor: probabilityColor(probability) }}
                key={`${tokenId}-${index}`}
                title={`Token ${tokenId}: ${probabilityToPercent(probability)}`}
              >
                <span className="token-chip-text">
                  {showTokenIds ? tokenId : visibleTokenText(tokenTexts[index] ?? "")}
                </span>
                <span className="token-chip-prob">{probabilityToPercent(probability)}</span>
              </span>
            );
          })}
        </div>
      ) : (
        <pre className={showTokenIds ? "completion-output ids" : "completion-output"}>
          {showTokenIds ? idOutput : decodedText}
        </pre>
      )}
    </section>
  );
}
