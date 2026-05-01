# AI Explorer Design Spec

This directory is intended to become the root of a fresh TypeScript/React repo that reimplements the existing Python `main.py --gui` experience in-browser.

The existing Python app remains the behavioral reference. The first web version should implement only the current GUI surface, using `Qwen/Qwen2.5-0.5B` via the browser-ready ONNX artifact. A later extension may add the `-Instruct` variant and chat-turn delimiter handling, but that is deliberately out of scope for v1.

## Goals

- Run fully in the browser with WebGPU acceleration.
- Match the current GUI workflow:
  - edit a base prompt
  - view the current prompt/completion
  - view top next-token candidates and probabilities
  - append a selected token
  - sample one token from the visible candidates
  - continue sampling until cancelled or EOS
  - delete generated tokens back to the editable base prompt
  - show token ids instead of token text
  - show probability coloring for each token in the prompt
- Use the existing Python behavior as a spec, not as shared code.
- Keep the app structured enough that inference backends, model variants, and UI can evolve independently.

## Non-Goals For V1

- No server-side inference.
- No multi-prompt branching UI from the TUI. The current NiceGUI `--gui` path does not expose this.
- No `Qwen2.5-0.5B-Instruct` chat-template handling yet.
- No persistence of prompts/history yet.
- No mobile-optimized interaction beyond a responsible loading gate and readable layout.
- No attempt to reproduce the Python UI pixel-for-pixel.

## Behavioral Reference

Current Python defaults from `config.toml`:

- Model: `Qwen/Qwen2.5-0.5B`
- Example prompt: `Once upon a time, there was a`
- Next tokens shown: `30`

Core Python concepts to mirror:

- `Explorer.set_prompt(text)` encodes text into token ids.
- `Explorer.get_top_n_tokens(n)` runs a forward pass and softmaxes the final-position logits.
- `Explorer.get_prompt_token_probabilities()` runs a forward pass and scores each actual prompt token given its previous context. First token uses `0.5`.
- `Explorer.append_token(token_id)` appends an arbitrary token id and decodes the full token list back to text.
- `Explorer.pop_token()` removes the last token and decodes the remaining ids.
- `TokenSession.append_weighted_token()` samples only from the currently displayed top-N tokens, weighted by their shown probabilities.
- NiceGUI's base-prompt behavior:
  - the text area is editable only while no generated completion tokens exist
  - once a token is appended, the base prompt is locked
  - Delete removes generated tokens only down to the base token count
  - Delete All resets to the base prompt

## Model And Inference

### Primary Model

Use:

```text
onnx-community/Qwen2.5-0.5B
```

This is the browser-ready ONNX/Transformers.js version of:

```text
Qwen/Qwen2.5-0.5B
```

Use the non-instruct base model because it matches the existing Python default and avoids chat-template delimiter behavior.

### Inference Library

Use `@huggingface/transformers` with WebGPU:

```ts
import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";
```

Load with:

```ts
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
  progress_callback,
});

const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
  device: "webgpu",
  dtype: "q4",
  progress_callback,
});
```

The implementation should use lower-level model calls rather than `pipeline("text-generation")`, because the app needs logits and probabilities, not only generated text.

### Inference Worker

Run all model loading and inference inside a dedicated Web Worker.

Reasons:

- Model load and forward passes must not block React rendering.
- The worker can own the tokenizer/model instances.
- Later backends can be swapped without contaminating UI state.

Use a simple typed message protocol first. Consider `comlink` only if the native protocol becomes noisy.

Worker responsibilities:

- load model and tokenizer
- emit model download/load progress
- encode prompt text
- decode token ids
- decode single candidate token ids
- run forward pass
- compute final-position top-N probabilities
- compute prompt-token probabilities
- append/delete/reset state if state is centralized in the worker

Recommended v1 state split:

- Worker owns inference objects only.
- Main thread owns app/session state as token ids and base prompt metadata.
- Main thread sends token arrays to worker when probabilities are needed.

This keeps UI state inspectable and testable without a model loaded.

### Inference Adapter API

Create a narrow TypeScript interface that the UI uses:

```ts
export interface TokenCandidate {
  tokenId: number;
  text: string;
  probability: number;
}

export interface TokenScoredText {
  tokenId: number;
  text: string;
  probability: number;
}

export interface InferenceBackend {
  load(): Promise<void>;
  encode(text: string): Promise<number[]>;
  decode(tokenIds: number[]): Promise<string>;
  decodeToken(tokenId: number): Promise<string>;
  getTopNextTokens(tokenIds: number[], n: number): Promise<TokenCandidate[]>;
  getPromptTokenProbabilities(tokenIds: number[]): Promise<number[]>;
  getEosTokenId(): Promise<number | null>;
}
```

Implementation notes:

- Use stable numerics for softmax:
  - subtract max logit
  - exponentiate
  - sum
  - divide
- For top-N:
  - avoid sorting the whole vocab when possible
  - a small min-heap or partial selection is fine
  - full sort is acceptable for a first pass if performance is adequate
- For prompt-token probabilities:
  - run one forward pass over the whole token sequence
  - for token position `i > 0`, use logits at `i - 1` and score token id at `i`
  - return `0.5` for the first token

### Caching

Rely on Transformers.js/browser caching for model files.

The first load should clearly expose progress. Subsequent visits should be faster if files are cached.

## Startup Flow

### Desktop-Likely User Agents

If the user agent suggests a desktop browser:

1. Render the app shell immediately.
2. Show a model-loading overlay with a determinate progress bar when possible.
3. Start downloading/loading model weights immediately.
4. Enable the app once tokenizer/model are ready.

Desktop-likely heuristic:

- no obvious mobile UA markers: `Mobi`, `Android`, `iPhone`, `iPad`, `iPod`
- screen width at least `900px`
- device memory, if available, at least `4GB`

Do not rely on this as a security or capability check. It is a UX heuristic.

### Non-Desktop Or Uncertain User Agents

If the UA suggests mobile/tablet/low-memory/uncertain:

1. Render the app shell in disabled state.
2. Show a modal warning that the app needs a large local model download and WebGPU support.
3. Offer:
   - Continue and download
   - Cancel
4. Only call `load()` after explicit user consent.

Suggested modal copy can be written during implementation, but keep it plain and brief.

### Failure States

Handle:

- WebGPU unavailable
- model download fails
- model load fails
- inference worker crashes
- out-of-memory or WebGPU device lost

Each should produce a recoverable UI state with a Retry action. For v1, no CPU fallback is required.

## Application State

Use a small Zustand store or React reducer. Recommendation: Zustand, because the app has shared state across controls, prompt display, token table, and async inference flows.

Core state:

```ts
type LoadStatus = "idle" | "needs-consent" | "loading" | "ready" | "error";

interface TokenExplorerState {
  loadStatus: LoadStatus;
  loadProgress: number | null;
  loadMessage: string | null;
  error: string | null;

  basePromptText: string;
  baseTokenIds: number[];
  tokenIds: number[];
  tokenTexts: string[];
  tokenProbabilities: number[];

  nextTokens: TokenCandidate[];
  selectedTokenId: number | null;

  showTokenIds: boolean;
  showProbabilities: boolean;
  isContinuing: boolean;
  eosTokenId: number | null;
}
```

Derived state:

- `completionTokenCount = tokenIds.length - baseTokenIds.length`
- `canEditPrompt = completionTokenCount === 0 && !showTokenIds && !isContinuing`
- `canDelete = completionTokenCount > 0 && !isContinuing`
- `canGenerate = tokenIds.length > 0 && !isContinuing && lastTokenId !== eosTokenId`

## Major Components

### `App`

Owns app bootstrapping and layout.

Responsibilities:

- initialize worker/backend
- run desktop/mobile loading gate
- route loading/error/ready states
- provide global keyboard handlers if added later

### `ModelLoadGate`

The startup overlay/modal.

States:

- desktop auto-load with progress
- mobile/uncertain consent prompt
- error with retry

### `PromptEditor`

Text area for the base prompt.

Responsibilities:

- display and edit base prompt
- debounce re-encoding/re-scoring while editable
- disabled once completion exists
- disabled in token-id mode

### `CompletionView`

Displays decoded prompt/completion output.

Modes:

- normal text mode: show decoded full text
- token id mode: show numeric ids separated by spaces
- probability mode: render token chips with probability-colored backgrounds

### `NextTokenTable`

Displays top-N next-token candidates.

Columns:

- token text or token id
- probability, with bar visualization

Interactions:

- click token row to append exact token
- selected/hover/focus states for future keyboard navigation

### `ControlsPanel`

Controls:

- Edit
- Next
- Delete
- Delete All
- Continue/Cancel
- Show token numbers
- Show probabilities

Use icon+text buttons where clarity matters, with `lucide-react` icons.

### `ProbabilityLegend`

Visible only when probability mode is enabled.

Shows 0, 25, 50, 75, 100 percent color swatches.

### `StatusBar`

Compact bottom or top-right status:

- model name
- WebGPU status
- token count
- load/inference activity

## Main Flows

### Initial Load

1. App starts.
2. Determine startup mode from UA/screen/device memory.
3. If desktop-likely, call `backend.load()` immediately.
4. If uncertain, wait for user consent.
5. During loading, render progress.
6. After load:
   - encode default prompt
   - decode token text pieces
   - compute top next tokens
   - compute prompt token probabilities if needed or lazily on first toggle
   - enable UI

### Editing Prompt

1. User edits text area while no generated completion exists.
2. Debounce input by about 200 ms.
3. Encode prompt.
4. Set `baseTokenIds = tokenIds = encoded`.
5. Refresh decoded token texts and next-token table.
6. If probability overlay is active, recompute prompt probabilities.

### Append Exact Token

1. User clicks a candidate in `NextTokenTable`.
2. Add candidate `tokenId` to `tokenIds`.
3. Decode full text and new token text.
4. Refresh top next tokens.
5. If probability overlay is active, compute probability for the appended token.
   - Simple v1: recompute all prompt token probabilities.
   - Later optimization: use previous final logits to score the appended token.

### Next

1. Sample one token from `nextTokens`, weighted by visible probabilities.
2. Append it using the same flow as exact-token append.

Important: This intentionally mirrors Python behavior, which samples only from displayed top tokens, not the full model distribution.

### Continue

1. Set `isContinuing = true`.
2. Loop:
   - if cancelled, stop
   - if last token is EOS, stop
   - sample visible weighted token
   - append token
   - allow React to paint between steps
3. Set `isContinuing = false`.

Use an `AbortController` or monotonic run id to cancel safely.

### Delete

1. If `tokenIds.length > baseTokenIds.length`, remove the last token.
2. Refresh decoded text and next-token table.
3. Stop at base token count.

### Delete All

1. Reset `tokenIds = baseTokenIds`.
2. Refresh decoded text and next-token table.

### Toggle Token Ids

1. If enabled:
   - disable prompt editing
   - render ids in output and candidate table
2. If disabled:
   - render decoded text again

### Toggle Probabilities

1. If enabled and probabilities are stale, compute prompt-token probabilities.
2. Render token chips colored by probability.
3. Show legend.
4. If disabled, render normal output.

## UI Direction

The current GUI uses playful pastel energy: warm yellow, pale blue, mint, amber, cyan, and deep blue ink. The web app should keep that warmth but feel more polished and deliberate.

Recommended palette:

- Ink: `#17233f`
- Muted ink: `#5b6680`
- Surface: `#ffffff`
- Page base: `#f8fbff`
- Warm wash: `#fff2c7`
- Cool wash: `#dff3ff`
- Mint wash: `#e4fae6`
- Amber accent: `#f6b73c`
- Cyan accent: `#1ca7c9`
- Positive/probability high: blue-green
- Probability low: warm peach

Background:

- Use a subtle linear or conic gradient wash across the full page.
- Avoid decorative blobs/orbs.
- Use quiet texture or layered flat color bands if needed.

Typography:

- Body/UI: `Inter` or `Instrument Sans`
- Display/title: `Fraunces` or `Instrument Serif`
- Code/token ids: `JetBrains Mono`

Prefer local font packages such as `@fontsource/inter`, `@fontsource/fraunces`, and `@fontsource/jetbrains-mono` to avoid runtime font network dependency.

Layout:

- Desktop-first workbench:
  - left/main column: prompt editor, completion, next-token table
  - right fixed-width rail: controls, legend, status
- Use restrained panels with radius <= 8px.
- Do not nest cards inside cards.
- Keep token table dense enough to scan 30 options.
- Candidate rows should have stable height and no layout shift on hover.

## Accessibility

- All controls keyboard reachable.
- Buttons have visible focus styles.
- Probability colors must not be the only signal: include numeric percentages.
- Continue/Cancel state must be announced through button text and `aria-busy`.
- Loading progress uses `<progress>` or ARIA progressbar.

## Suggested Libraries

Runtime:

- `react`
- `react-dom`
- `@huggingface/transformers`
- `zustand`
- `lucide-react`

Build/test:

- `vite`
- `typescript`
- `vitest`
- `@testing-library/react`
- `@testing-library/user-event`
- `playwright` for browser smoke tests

Styling:

- plain CSS modules or scoped CSS files
- CSS custom properties for palette
- no Tailwind required for v1

Recommended package manager:

- `npm`, unless the repo later standardizes on another JS package manager

## Proposed Directory Structure

```text
web-token-explorer/
  DESIGN_SPEC.md
  package.json
  index.html
  vite.config.ts
  tsconfig.json
  src/
    main.tsx
    App.tsx
    styles/
      theme.css
      app.css
    components/
      ModelLoadGate.tsx
      PromptEditor.tsx
      CompletionView.tsx
      NextTokenTable.tsx
      ControlsPanel.tsx
      ProbabilityLegend.tsx
      StatusBar.tsx
    inference/
      types.ts
      workerClient.ts
      inference.worker.ts
      logits.ts
      sampling.ts
    state/
      explorerStore.ts
    utils/
      device.ts
      probabilityColor.ts
```

## Testing Plan

Unit tests without loading the model:

- softmax/top-k utility
- weighted visible-token sampling
- probability color mapping
- derived state logic
- device/UA load-gate heuristic

Integration tests with mocked inference backend:

- startup consent vs auto-load behavior
- prompt edit updates token state
- append token locks editor
- delete unlocks editor at base prompt
- continue can be cancelled
- token id mode disables editing
- probability mode requests scoring

Manual/browser smoke test with real model:

- desktop Chrome or Edge with WebGPU
- first-load progress appears
- model loads from `onnx-community/Qwen2.5-0.5B`
- default prompt produces 30 next tokens
- clicking a candidate appends exactly that token id
- Next samples from visible candidates
- Delete and Delete All match Python GUI behavior
- probability overlay colors prompt tokens

## Implementation Phases

### Phase 1: Scaffold

- Vite React TypeScript app
- theme and layout shell
- mocked inference backend
- all core components wired to mocked data

### Phase 2: Real Inference

- Web Worker
- Transformers.js model/tokenizer load
- progress reporting
- encode/decode
- final logits top-N
- prompt-token probabilities

### Phase 3: UX Polish

- loading gate details
- error/retry states
- Continue cancellation
- probability overlay performance
- responsive desktop/tablet behavior

### Phase 4: Verification

- unit tests
- mocked integration tests
- Playwright smoke test with a real browser
- compare a few prompts against Python/PyTorch top tokens and probabilities

## Future Extension: Instruct Variant

Potential model:

```text
onnx-community/Qwen2.5-0.5B-Instruct
```

Future additions:

- model selector: Base vs Instruct
- chat template handling
- user/assistant role delimiters
- visual distinction between system/user/assistant/generated tokens
- option to inspect raw delimiter/control tokens

Do not design v1 state around chat messages yet. Keep v1 token-first so the base-model behavior stays simple and faithful.
