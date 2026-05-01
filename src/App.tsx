import { useCallback, useEffect, useMemo, useRef } from "react";
import { AlertTriangle } from "lucide-react";
import { CompletionView } from "./components/CompletionView";
import { ControlsPanel } from "./components/ControlsPanel";
import { ModelLoadGate } from "./components/ModelLoadGate";
import { NextTokenTable } from "./components/NextTokenTable";
import { ProbabilityLegend } from "./components/ProbabilityLegend";
import { PromptEditor } from "./components/PromptEditor";
import { DEFAULT_TOP_N, MODEL_ID } from "./inference/types";
import type { InferenceBackend, TokenCandidate } from "./inference/types";
import { sampleVisibleToken } from "./inference/sampling";
import { WorkerInferenceBackend } from "./inference/workerClient";
import {
  getDerivedExplorerState,
  useExplorerStore,
} from "./state/explorerStore";
import { currentDeviceLoadDecision, getWebGpuStatus } from "./utils/device";
import aiExplorerTitleUrl from "./assets/ai-explorer.svg";

const PROMPT_DEBOUNCE_MS = 220;
const PAINT_DELAY_MS = 35;
const PROJECT_GITHUB_URL = "https://github.com/simonchatts/ai-explorer";
const TOKEN_EXPLORER_URL = "https://github.com/willkurt/token-explorer";
const MODEL_HF_URL = "https://huggingface.co/onnx-community/Qwen2.5-0.5B";

function isEditableShortcutTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return Boolean(
    target.closest("input, textarea, select, button, [contenteditable='true']"),
  );
}

function waitForPaint(): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, PAINT_DELAY_MS);
  });
}

function InvertocatLogo() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" width="20" height="20" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.62 7.62 0 0 1 8 3.86c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

export function App() {
  const state = useExplorerStore();
  const setStatePatch = useExplorerStore((store) => store.setStatePatch);
  const derived = useMemo(() => getDerivedExplorerState(state), [state]);
  const backendRef = useRef<InferenceBackend | null>(null);
  const promptRunId = useRef(0);
  const actionRunId = useRef(0);
  const continueRunId = useRef(0);
  const lastEncodedPrompt = useRef<string | null>(null);
  const promptTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const deviceDecision = useMemo(() => currentDeviceLoadDecision(), []);
  const webGpuStatus = useMemo(() => getWebGpuStatus(), []);

  useEffect(() => {
    const backend = new WorkerInferenceBackend(
      (progress) =>
        setStatePatch({
          loadProgress: progress.progress,
          loadMessage: progress.message,
        }),
      (error) =>
        setStatePatch({
          loadStatus: "error",
          error,
          isInferring: false,
          isContinuing: false,
        }),
    );
    backendRef.current = backend;

    return () => {
      backend.terminate();
      backendRef.current = null;
    };
  }, [setStatePatch]);

  const refreshForTokens = useCallback(
    async (
      tokenIds: number[],
      options: { baseTokenIds?: number[]; basePromptText?: string } = {},
    ) => {
      const backend = backendRef.current;
      if (!backend) throw new Error("Inference backend is not ready.");

      const [decodedText, tokenTexts, nextTokens, tokenProbabilities] =
        await Promise.all([
          backend.decode(tokenIds),
          Promise.all(tokenIds.map((tokenId) => backend.decodeToken(tokenId))),
          tokenIds.length > 0
            ? backend.getTopNextTokens(tokenIds, DEFAULT_TOP_N)
            : Promise.resolve([]),
          state.showProbabilities
            ? backend.getPromptTokenProbabilities(tokenIds)
            : Promise.resolve(state.tokenProbabilities),
        ]);

      setStatePatch({
        decodedText,
        tokenIds,
        tokenTexts,
        nextTokens,
        tokenProbabilities,
        probabilitiesStale: !state.showProbabilities,
        selectedTokenId: null,
        ...(options.baseTokenIds ? { baseTokenIds: options.baseTokenIds } : {}),
        ...(options.basePromptText != null
          ? { basePromptText: options.basePromptText }
          : {}),
      });
    },
    [setStatePatch, state.showProbabilities, state.tokenProbabilities],
  );

  const initializePrompt = useCallback(async () => {
    const backend = backendRef.current;
    if (!backend) throw new Error("Inference backend is not ready.");
    const encoded = await backend.encode(state.basePromptText);
    lastEncodedPrompt.current = state.basePromptText;
    await refreshForTokens(encoded, {
      baseTokenIds: encoded,
      basePromptText: state.basePromptText,
    });
  }, [refreshForTokens, state.basePromptText]);

  const loadModel = useCallback(async () => {
    const backend = backendRef.current;
    if (!backend) return;

    setStatePatch({
      loadStatus: "loading",
      loadProgress: null,
      loadMessage: "Starting local model load",
      error: null,
      isInferring: true,
    });

    try {
      await backend.load();
      const eosTokenId = await backend.getEosTokenId();
      setStatePatch({
        loadStatus: "ready",
        eosTokenId,
        loadProgress: 100,
        loadMessage: "Model ready",
      });
      await initializePrompt();
      setStatePatch({ isInferring: false });
    } catch (error) {
      setStatePatch({
        loadStatus: "error",
        error: error instanceof Error ? error.message : String(error),
        isInferring: false,
        isContinuing: false,
      });
    }
  }, [initializePrompt, setStatePatch]);

  useEffect(() => {
    if (state.loadStatus !== "idle") return;

    if (webGpuStatus === "unavailable") {
      setStatePatch({
        loadStatus: "error",
        error:
          "WebGPU is not available in this browser. Try desktop Chrome or Edge with WebGPU enabled.",
      });
      return;
    }

    if (deviceDecision.shouldAutoLoad) {
      void loadModel();
    } else {
      setStatePatch({
        loadStatus: "needs-consent",
        loadMessage: deviceDecision.reason,
      });
    }
  }, [
    deviceDecision.reason,
    deviceDecision.shouldAutoLoad,
    loadModel,
    setStatePatch,
    state.loadStatus,
    webGpuStatus,
  ]);

  useEffect(() => {
    if (state.loadStatus !== "ready" || !derived.canEditPrompt) return;
    if (state.basePromptText === lastEncodedPrompt.current) return;

    const runId = promptRunId.current + 1;
    promptRunId.current = runId;
    const timer = window.setTimeout(async () => {
      const backend = backendRef.current;
      if (!backend || promptRunId.current !== runId) return;

      setStatePatch({ isInferring: true, error: null });
      try {
        const encoded = await backend.encode(state.basePromptText);
        if (promptRunId.current !== runId) return;
        lastEncodedPrompt.current = state.basePromptText;
        await refreshForTokens(encoded, {
          baseTokenIds: encoded,
          basePromptText: state.basePromptText,
        });
      } catch (error) {
        setStatePatch({
          loadStatus: "error",
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        if (promptRunId.current === runId) {
          setStatePatch({ isInferring: false });
        }
      }
    }, PROMPT_DEBOUNCE_MS);

    return () => window.clearTimeout(timer);
  }, [
    derived.canEditPrompt,
    refreshForTokens,
    setStatePatch,
    state.basePromptText,
    state.loadStatus,
  ]);

  const runTokenAction = useCallback(
    async (nextTokenIds: number[]) => {
      const runId = actionRunId.current + 1;
      actionRunId.current = runId;
      setStatePatch({ isInferring: true, error: null });

      try {
        await refreshForTokens(nextTokenIds);
      } catch (error) {
        setStatePatch({
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        if (actionRunId.current === runId) {
          setStatePatch({ isInferring: false });
        }
      }
    },
    [refreshForTokens, setStatePatch],
  );

  const appendCandidate = useCallback(
    async (candidate: TokenCandidate) => {
      if (state.isContinuing || state.isInferring) return;
      setStatePatch({ selectedTokenId: candidate.tokenId });
      await runTokenAction([...state.tokenIds, candidate.tokenId]);
    },
    [
      runTokenAction,
      setStatePatch,
      state.isContinuing,
      state.isInferring,
      state.tokenIds,
    ],
  );

  const appendSampled = useCallback(async () => {
    const sampled = sampleVisibleToken(state.nextTokens);
    if (!sampled) return;
    await appendCandidate(sampled);
  }, [appendCandidate, state.nextTokens]);

  const deleteOne = useCallback(async () => {
    if (!derived.canDelete) return;
    await runTokenAction(state.tokenIds.slice(0, -1));
  }, [derived.canDelete, runTokenAction, state.tokenIds]);

  const deleteAll = useCallback(async () => {
    if (
      state.tokenIds.length === state.baseTokenIds.length ||
      state.isInferring
    )
      return;
    await runTokenAction([...state.baseTokenIds]);
  }, [
    runTokenAction,
    state.baseTokenIds,
    state.isInferring,
    state.tokenIds.length,
  ]);

  const toggleProbabilities = useCallback(async () => {
    const nextValue = !state.showProbabilities;
    setStatePatch({ showProbabilities: nextValue });
    if (!nextValue || !state.probabilitiesStale) return;

    const backend = backendRef.current;
    if (!backend) return;
    setStatePatch({ isInferring: true });
    try {
      const tokenProbabilities = await backend.getPromptTokenProbabilities(
        state.tokenIds,
      );
      setStatePatch({ tokenProbabilities, probabilitiesStale: false });
    } catch (error) {
      setStatePatch({
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setStatePatch({ isInferring: false });
    }
  }, [
    setStatePatch,
    state.probabilitiesStale,
    state.showProbabilities,
    state.tokenIds,
  ]);

  const startContinue = useCallback(async () => {
    if (!derived.canGenerate || state.nextTokens.length === 0) return;

    const runId = continueRunId.current + 1;
    continueRunId.current = runId;
    setStatePatch({ isContinuing: true, isInferring: true, error: null });

    try {
      let currentTokenIds = [...state.tokenIds];
      let currentCandidates = [...state.nextTokens];
      const backend = backendRef.current;
      if (!backend) throw new Error("Inference backend is not ready.");

      while (continueRunId.current === runId) {
        const last = currentTokenIds[currentTokenIds.length - 1];
        if (state.eosTokenId != null && last === state.eosTokenId) break;

        const sampled = sampleVisibleToken(currentCandidates);
        if (!sampled) break;
        currentTokenIds = [...currentTokenIds, sampled.tokenId];

        const [decodedText, tokenTexts, nextTokens, tokenProbabilities] =
          await Promise.all([
            backend.decode(currentTokenIds),
            Promise.all(
              currentTokenIds.map((tokenId) => backend.decodeToken(tokenId)),
            ),
            backend.getTopNextTokens(currentTokenIds, DEFAULT_TOP_N),
            state.showProbabilities
              ? backend.getPromptTokenProbabilities(currentTokenIds)
              : Promise.resolve(state.tokenProbabilities),
          ]);

        if (continueRunId.current !== runId) break;

        currentCandidates = nextTokens;
        setStatePatch({
          decodedText,
          tokenIds: currentTokenIds,
          tokenTexts,
          nextTokens,
          tokenProbabilities,
          probabilitiesStale: !state.showProbabilities,
          selectedTokenId: sampled.tokenId,
        });

        await waitForPaint();
      }
    } catch (error) {
      setStatePatch({
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      if (continueRunId.current === runId) {
        setStatePatch({
          isContinuing: false,
          isInferring: false,
          selectedTokenId: null,
        });
      }
    }
  }, [
    derived.canGenerate,
    setStatePatch,
    state.eosTokenId,
    state.nextTokens,
    state.showProbabilities,
    state.tokenIds,
    state.tokenProbabilities,
  ]);

  const cancelContinue = useCallback(() => {
    continueRunId.current += 1;
    setStatePatch({
      isContinuing: false,
      isInferring: false,
      selectedTokenId: null,
    });
  }, [setStatePatch]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey)
        return;

      if (state.isContinuing && (event.key === " " || event.key === "Escape")) {
        event.preventDefault();
        cancelContinue();
        return;
      }

      if (isEditableShortcutTarget(event.target)) return;

      if (event.key === "ArrowRight" || event.key === "Enter") {
        if (!derived.canGenerate) return;
        event.preventDefault();
        void appendSampled();
        return;
      }

      if (event.key === "ArrowLeft" || event.key === "Backspace") {
        if (!derived.canDelete) return;
        event.preventDefault();
        void deleteOne();
        return;
      }

      if (event.key === " ") {
        if (!derived.canGenerate) return;
        event.preventDefault();
        void startContinue();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    appendSampled,
    cancelContinue,
    deleteOne,
    derived.canDelete,
    derived.canGenerate,
    startContinue,
    state.isContinuing,
  ]);

  const handleEdit = useCallback(() => {
    if (state.tokenIds.length > state.baseTokenIds.length) {
      void deleteAll();
      return;
    }
    promptTextareaRef.current?.focus();
  }, [deleteAll, state.baseTokenIds.length, state.tokenIds.length]);

  const retry = useCallback(() => {
    void loadModel();
  }, [loadModel]);

  return (
    <main
      className="app-shell"
      aria-busy={state.isInferring || state.isContinuing}
    >
      <header className="app-header">
        <div>
          <h1 className="title-logo-heading">
            <img
              className="title-logo"
              src={aiExplorerTitleUrl}
              alt="AI Explorer"
            />
          </h1>
        </div>
        <a
          className="github-link"
          href={PROJECT_GITHUB_URL}
          target="_blank"
          rel="noreferrer"
          aria-label="Open ai-explorer on GitHub"
          title="Open ai-explorer on GitHub"
        >
          <InvertocatLogo />
        </a>
      </header>

      <section
        className="workbench"
        aria-disabled={state.loadStatus !== "ready"}
      >
        <div className="main-column">
          <PromptEditor
            ref={promptTextareaRef}
            value={state.basePromptText}
            disabled={!derived.canEditPrompt}
            onChange={(basePromptText) => setStatePatch({ basePromptText })}
          />
          <CompletionView
            decodedText={state.decodedText}
            tokenIds={state.tokenIds}
            tokenTexts={state.tokenTexts}
            tokenProbabilities={state.tokenProbabilities}
            baseTokenCount={state.baseTokenIds.length}
            showTokenIds={state.showTokenIds}
            showProbabilities={state.showProbabilities}
          />
          <NextTokenTable
            candidates={state.nextTokens}
            selectedTokenId={state.selectedTokenId}
            showTokenIds={state.showTokenIds}
            showProbabilities={state.showProbabilities}
            disabled={state.isContinuing || !derived.canGenerate}
            onAppend={appendCandidate}
          />
        </div>

        <aside className="side-rail">
          {state.error ? (
            <div className="inline-error" role="alert">
              <AlertTriangle size={18} aria-hidden="true" />
              <span>{state.error}</span>
            </div>
          ) : null}
          <ControlsPanel
            canGenerate={derived.canGenerate}
            canDelete={derived.canDelete}
            isContinuing={state.isContinuing}
            isInferring={state.isInferring}
            showTokenIds={state.showTokenIds}
            showProbabilities={state.showProbabilities}
            onEdit={handleEdit}
            onNext={appendSampled}
            onDelete={deleteOne}
            onDeleteAll={deleteAll}
            onContinue={state.isContinuing ? cancelContinue : startContinue}
            onToggleTokenIds={() =>
              setStatePatch({ showTokenIds: !state.showTokenIds })
            }
            onToggleProbabilities={toggleProbabilities}
          />
          <ProbabilityLegend visible={state.showProbabilities} />
        </aside>
      </section>

      <footer className="app-footer">
        <span>
          Model:{" "}
          <a href={MODEL_HF_URL} target="_blank" rel="noreferrer">
            {MODEL_ID}
          </a>
        </span>
        <span>
          Code at{" "}
          <a href={PROJECT_GITHUB_URL} target="_blank" rel="noreferrer">
            simonchatts/ai-explorer
          </a>
        </span>
        <span>
          Originally based on{" "}
          <a href={TOKEN_EXPLORER_URL} target="_blank" rel="noreferrer">
            willkurt/token-explorer
          </a>
        </span>
      </footer>

      <ModelLoadGate
        status={state.loadStatus}
        progress={state.loadProgress}
        message={state.loadMessage}
        error={state.error}
        consentReason={deviceDecision.reason}
        onContinue={loadModel}
        onCancel={() =>
          setStatePatch({
            loadStatus: "needs-consent",
            loadMessage:
              "Model download cancelled. The workbench will stay disabled until you continue.",
          })
        }
        onRetry={retry}
      />
    </main>
  );
}
