import { describe, expect, it } from "vitest";
import { getDerivedExplorerState, initialExplorerState } from "./explorerStore";

describe("derived explorer state", () => {
  it("allows editing only when there are no generated tokens", () => {
    const state = {
      ...initialExplorerState,
      loadStatus: "ready" as const,
      baseTokenIds: [10, 20],
      tokenIds: [10, 20],
    };

    expect(getDerivedExplorerState(state).canEditPrompt).toBe(true);
    expect(
      getDerivedExplorerState({
        ...state,
        tokenIds: [10, 20, 30],
      }).canEditPrompt,
    ).toBe(false);
  });

  it("stops generation at EOS", () => {
    const state = {
      ...initialExplorerState,
      loadStatus: "ready" as const,
      eosTokenId: 99,
      baseTokenIds: [10],
      tokenIds: [10, 99],
    };

    expect(getDerivedExplorerState(state).canGenerate).toBe(false);
  });
});
