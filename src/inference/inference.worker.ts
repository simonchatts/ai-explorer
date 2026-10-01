import { AutoModelForCausalLM, AutoTokenizer, Tensor } from "@huggingface/transformers";
import { MODEL_ID } from "./types";
import { topTokensFromLogits } from "./logits";
import { InferenceCache } from "./inferenceCache";
import type { ForwardInputs, ForwardOutputs } from "./inferenceCache";

type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
type Model = Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>>;

type WorkerRequest =
  | { id: number; type: "load" }
  | { id: number; type: "encode"; text: string }
  | { id: number; type: "decode"; tokenIds: number[] }
  | { id: number; type: "decodeToken"; tokenId: number }
  | { id: number; type: "getTokenPredictions"; tokenIds: number[]; n: number; includeProbabilities: boolean }
  | { id: number; type: "getPromptTokenProbabilities"; tokenIds: number[] }
  | { id: number; type: "getEosTokenId" };

let tokenizer: Tokenizer | null = null;
let model: Model | null = null;
let loadPromise: Promise<void> | null = null;

function postResult(id: number, result: unknown): void {
  self.postMessage({ id, type: "result", result });
}

function postError(id: number, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  self.postMessage({ id, type: "error", error: message });
}

function postProgress(message: string, progress: number | null = null): void {
  self.postMessage({ type: "progress", progress: { message, progress } });
}

function toNumberArray(value: unknown): number[] {
  if (Array.isArray(value)) return value.map(Number);
  if (value && typeof value === "object" && "data" in value) {
    return Array.from((value as { data: Iterable<number | bigint> }).data, Number);
  }
  if (ArrayBuffer.isView(value) && "length" in value) {
    return Array.from(value as unknown as ArrayLike<number | bigint>, Number);
  }
  return [];
}

function createInt64Tensor(values: ArrayLike<number>, dims: number[]): Tensor {
  const data = BigInt64Array.from(values, (value) => BigInt(value));
  return new Tensor("int64", data, dims);
}

function createModelInputs({ tokenIds, positionOffset, attentionLength, pastKeyValues }: ForwardInputs<Tensor>) {
  const positions = tokenIds.map((_, index) => positionOffset + index);
  return {
    input_ids: createInt64Tensor(tokenIds, [1, tokenIds.length]),
    attention_mask: createInt64Tensor(Array.from({ length: attentionLength }, () => 1), [1, attentionLength]),
    position_ids: createInt64Tensor(positions, [1, tokenIds.length]),
    past_key_values: pastKeyValues,
  };
}

const inferenceCache = new InferenceCache<Tensor>(async (inputs) => {
  if (!model) throw new Error("Model has not loaded.");
  const outputs = await model(createModelInputs(inputs));
  return {
    logits: outputs.logits as ForwardOutputs<Tensor>["logits"],
    // Let the cache own disposal of old KV tensors once the forward pass finishes.
    pastKeyValues: model.getPastKeyValues(outputs, null) as Record<string, Tensor>,
  };
});

async function ensureLoaded(): Promise<void> {
  if (tokenizer && model) return;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    if (!("gpu" in navigator)) {
      throw new Error("WebGPU is not available in this browser.");
    }

    const progressCallback = (event: { status?: string; file?: string; progress?: number }) => {
      const bits = [event.status, event.file].filter(Boolean).join(": ");
      postProgress(bits || "Loading model files", typeof event.progress === "number" ? event.progress : null);
    };

    postProgress("Loading tokenizer", null);
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      progress_callback: progressCallback,
    });

    postProgress("Loading model weights", null);
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      device: "webgpu",
      dtype: "q4",
      progress_callback: progressCallback,
    });

    postProgress("Model ready", 100);
  })();

  try {
    await loadPromise;
  } catch (error) {
    tokenizer = null;
    model = null;
    loadPromise = null;
    throw error;
  }
}

function decodeTokenSync(tokenId: number): string {
  if (!tokenizer) throw new Error("Tokenizer has not loaded.");
  return tokenizer.decode([tokenId], { skip_special_tokens: false });
}

self.addEventListener("message", async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  try {
    if (message.type === "load") {
      await ensureLoaded();
      postResult(message.id, undefined);
      return;
    }

    await ensureLoaded();
    if (!tokenizer) throw new Error("Tokenizer has not loaded.");

    if (message.type === "encode") {
      const encoded = await tokenizer(message.text, { add_special_tokens: false });
      postResult(message.id, toNumberArray(encoded.input_ids));
      return;
    }

    if (message.type === "decode") {
      postResult(message.id, tokenizer.decode(message.tokenIds, { skip_special_tokens: false }));
      return;
    }

    if (message.type === "decodeToken") {
      postResult(message.id, decodeTokenSync(message.tokenId));
      return;
    }

    if (message.type === "getTokenPredictions") {
      const prediction = await inferenceCache.infer(message.tokenIds, message.includeProbabilities);
      postResult(message.id, {
        nextTokens: prediction ? topTokensFromLogits(prediction.nextLogits, message.n, decodeTokenSync) : [],
        tokenProbabilities: message.includeProbabilities ? prediction?.tokenProbabilities ?? [] : null,
      });
      return;
    }

    if (message.type === "getPromptTokenProbabilities") {
      const prediction = await inferenceCache.infer(message.tokenIds, true);
      postResult(message.id, prediction?.tokenProbabilities ?? []);
      return;
    }

    if (message.type === "getEosTokenId") {
      const tokenizerWithConfig = tokenizer as {
        eos_token_id?: unknown;
        config?: { eos_token_id?: unknown };
      };
      const raw = tokenizerWithConfig.eos_token_id ?? tokenizerWithConfig.config?.eos_token_id ?? null;
      const value = Array.isArray(raw) ? raw[0] : raw;
      postResult(message.id, value == null ? null : Number(value));
    }
  } catch (error) {
    postError(message.id, error);
  }
});
