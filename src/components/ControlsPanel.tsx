import {
  ChevronsRight,
  Hash,
  Pause,
  Pencil,
  Play,
  RotateCcw,
  StepForward,
  Trash2,
} from "lucide-react";

interface ControlsPanelProps {
  canGenerate: boolean;
  canDelete: boolean;
  isContinuing: boolean;
  isInferring: boolean;
  showTokenIds: boolean;
  showProbabilities: boolean;
  onEdit: () => void;
  onNext: () => void;
  onDelete: () => void;
  onDeleteAll: () => void;
  onContinue: () => void;
  onToggleTokenIds: () => void;
  onToggleProbabilities: () => void;
}

export function ControlsPanel({
  canGenerate,
  canDelete,
  isContinuing,
  isInferring,
  showTokenIds,
  showProbabilities,
  onEdit,
  onNext,
  onDelete,
  onDeleteAll,
  onContinue,
  onToggleTokenIds,
  onToggleProbabilities,
}: ControlsPanelProps) {
  return (
    <section className="panel controls-panel" aria-labelledby="controls-title">
      <div className="panel-heading">
        <h2 id="controls-title">Controls</h2>
      </div>

      <div className="button-grid">
        <button type="button" className="tool-button" onClick={onEdit} disabled={isInferring || isContinuing}>
          <Pencil size={18} aria-hidden="true" />
          Edit
        </button>
        <button type="button" className="tool-button" onClick={onNext} disabled={!canGenerate}>
          <StepForward size={18} aria-hidden="true" />
          Next
        </button>
        <button type="button" className="tool-button" onClick={onDelete} disabled={!canDelete}>
          <Trash2 size={18} aria-hidden="true" />
          Delete
        </button>
        <button type="button" className="tool-button" onClick={onDeleteAll} disabled={!canDelete}>
          <RotateCcw size={18} aria-hidden="true" />
          Delete All
        </button>
      </div>

      <button
        type="button"
        className={isContinuing ? "primary-button danger" : "primary-button"}
        onClick={onContinue}
        disabled={!isContinuing && !canGenerate}
        aria-busy={isContinuing}
      >
        {isContinuing ? <Pause size={18} aria-hidden="true" /> : <ChevronsRight size={18} aria-hidden="true" />}
        {isContinuing ? "Cancel" : "Continue"}
      </button>

      <div className="toggle-stack">
        <label className="toggle-row">
          <input type="checkbox" checked={showTokenIds} onChange={onToggleTokenIds} />
          <span>
            <Hash size={17} aria-hidden="true" />
            Show token numbers
          </span>
        </label>
        <label className="toggle-row">
          <input type="checkbox" checked={showProbabilities} onChange={onToggleProbabilities} />
          <span>
            <Play size={17} aria-hidden="true" />
            Show probabilities
          </span>
        </label>
      </div>
    </section>
  );
}
