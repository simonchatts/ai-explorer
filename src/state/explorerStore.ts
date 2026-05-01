import { create } from "zustand";
import { DEFAULT_PROMPT } from "../inference/types";
import type { TokenCandidate } from "../inference/types";

export type LoadStatus = "idle" | "needs-consent" | "loading" | "ready" | "error";

export interface TokenExplorerState {
  loadStatus: LoadStatus;
  loadProgress: number | null;
  loadMessage: string | null;
  error: string | null;
  isInferring: boolean;

  basePromptText: string;
  decodedText: string;
  baseTokenIds: number[];
  tokenIds: number[];
  tokenTexts: string[];
  tokenProbabilities: number[];
  probabilitiesStale: boolean;

  nextTokens: TokenCandidate[];
  selectedTokenId: number | null;

  showTokenIds: boolean;
  showProbabilities: boolean;
  isContinuing: boolean;
  eosTokenId: number | null;
}

interface TokenExplorerActions {
  setStatePatch: (patch: Partial<TokenExplorerState>) => void;
  resetRuntime: () => void;
}

export type ExplorerStore = TokenExplorerState & TokenExplorerActions;

export interface DerivedExplorerState {
  completionTokenCount: number;
  canEditPrompt: boolean;
  canDelete: boolean;
  canGenerate: boolean;
}

export const initialExplorerState: TokenExplorerState = {
  loadStatus: "idle",
  loadProgress: null,
  loadMessage: null,
  error: null,
  isInferring: false,

  basePromptText: DEFAULT_PROMPT,
  decodedText: DEFAULT_PROMPT,
  baseTokenIds: [],
  tokenIds: [],
  tokenTexts: [],
  tokenProbabilities: [],
  probabilitiesStale: true,

  nextTokens: [],
  selectedTokenId: null,

  showTokenIds: false,
  showProbabilities: false,
  isContinuing: false,
  eosTokenId: null,
};

export function getDerivedExplorerState(state: TokenExplorerState): DerivedExplorerState {
  const completionTokenCount = Math.max(0, state.tokenIds.length - state.baseTokenIds.length);
  const lastTokenId = state.tokenIds[state.tokenIds.length - 1] ?? null;

  return {
    completionTokenCount,
    canEditPrompt:
      state.loadStatus === "ready" &&
      completionTokenCount === 0 &&
      !state.showTokenIds &&
      !state.isContinuing &&
      !state.isInferring,
    canDelete: completionTokenCount > 0 && !state.isContinuing && !state.isInferring,
    canGenerate:
      state.loadStatus === "ready" &&
      state.tokenIds.length > 0 &&
      !state.isContinuing &&
      !state.isInferring &&
      (state.eosTokenId == null || lastTokenId !== state.eosTokenId),
  };
}

export const useExplorerStore = create<ExplorerStore>((set) => ({
  ...initialExplorerState,
  setStatePatch: (patch) => set(patch),
  resetRuntime: () =>
    set({
      ...initialExplorerState,
      loadStatus: "idle",
    }),
}));
