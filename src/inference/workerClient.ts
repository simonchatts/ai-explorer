import type { InferenceBackend, LoadProgress, TokenCandidate } from "./types";

type WorkerRequest =
  | { id: number; type: "load" }
  | { id: number; type: "encode"; text: string }
  | { id: number; type: "decode"; tokenIds: number[] }
  | { id: number; type: "decodeToken"; tokenId: number }
  | { id: number; type: "getTopNextTokens"; tokenIds: number[]; n: number }
  | { id: number; type: "getPromptTokenProbabilities"; tokenIds: number[] }
  | { id: number; type: "getEosTokenId" };

type WorkerResponse =
  | { id: number; type: "result"; result: unknown }
  | { id: number; type: "error"; error: string }
  | { type: "progress"; progress: LoadProgress };

export class WorkerInferenceBackend implements InferenceBackend {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (value: unknown) => void; reject: (reason?: unknown) => void }
  >();

  constructor(
    private readonly onProgress: (progress: LoadProgress) => void,
    private readonly onCrash: (error: string) => void,
  ) {
    this.worker = new Worker(new URL("./inference.worker.ts", import.meta.url), { type: "module" });
    this.worker.addEventListener("message", this.handleMessage);
    this.worker.addEventListener("error", (event) => {
      this.rejectAll(event.message || "Inference worker crashed.");
      this.onCrash(event.message || "Inference worker crashed.");
    });
    this.worker.addEventListener("messageerror", () => {
      this.rejectAll("Inference worker sent an unreadable message.");
      this.onCrash("Inference worker sent an unreadable message.");
    });
  }

  terminate(): void {
    this.rejectAll("Inference worker was terminated.");
    this.worker.terminate();
  }

  load(): Promise<void> {
    return this.request<void>({ id: 0, type: "load" });
  }

  encode(text: string): Promise<number[]> {
    return this.request<number[]>({ id: 0, type: "encode", text });
  }

  decode(tokenIds: number[]): Promise<string> {
    return this.request<string>({ id: 0, type: "decode", tokenIds });
  }

  decodeToken(tokenId: number): Promise<string> {
    return this.request<string>({ id: 0, type: "decodeToken", tokenId });
  }

  getTopNextTokens(tokenIds: number[], n: number): Promise<TokenCandidate[]> {
    return this.request<TokenCandidate[]>({ id: 0, type: "getTopNextTokens", tokenIds, n });
  }

  getPromptTokenProbabilities(tokenIds: number[]): Promise<number[]> {
    return this.request<number[]>({ id: 0, type: "getPromptTokenProbabilities", tokenIds });
  }

  getEosTokenId(): Promise<number | null> {
    return this.request<number | null>({ id: 0, type: "getEosTokenId" });
  }

  private request<T>(message: WorkerRequest): Promise<T> {
    const id = this.nextId;
    this.nextId += 1;
    const request = { ...message, id } as WorkerRequest;

    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject,
      });
      this.worker.postMessage(request);
    });
  }

  private handleMessage = (event: MessageEvent<WorkerResponse>) => {
    const message = event.data;
    if (message.type === "progress") {
      this.onProgress(message.progress);
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);

    if (message.type === "error") {
      pending.reject(new Error(message.error));
    } else {
      pending.resolve(message.result);
    }
  };

  private rejectAll(reason: string): void {
    for (const pending of this.pending.values()) {
      pending.reject(new Error(reason));
    }
    this.pending.clear();
  }
}
