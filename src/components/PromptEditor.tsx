import { forwardRef } from "react";

interface PromptEditorProps {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
}

export const PromptEditor = forwardRef<HTMLTextAreaElement, PromptEditorProps>(
  ({ value, disabled, onChange }, ref) => (
    <section className="panel prompt-panel" aria-labelledby="prompt-title">
      <div className="panel-heading">
        <h2 id="prompt-title">Base Prompt</h2>
        <span>{disabled ? "Locked" : "Editable"}</span>
      </div>
      <textarea
        ref={ref}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        spellCheck={false}
        aria-label="Base prompt"
      />
    </section>
  ),
);

PromptEditor.displayName = "PromptEditor";
