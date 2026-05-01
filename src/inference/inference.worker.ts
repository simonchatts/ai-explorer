import { AutoModelForCausalLM, AutoTokenizer, Tensor } from "@huggingface/transformers";
import { MODEL_ID } from "./types";
import { probabilityForToken, topTokensFromLogits } from "./logits";

type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
type Model = Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>>;

type WorkerRequest =
  | { id: number; type: "load" }
  | { id: number; type: "encode"; text: string }
  | { id: number; type: "decode"; tokenIds: number[] }
  | { id: number; type: "decodeToken"; tokenId: number }
  | { id: number; type: "getTopNextTokens"; tokenIds: number[]; n: number }
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

function createModelInputs(tokenIds: number[]): {
  input_ids: Tensor;
  attention_mask: Tensor;
  position_ids: Tensor;
} {
  const sequenceLength = tokenIds.length;
  const dims = [1, sequenceLength];
  const positions = Array.from({ length: sequenceLength }, (_, index) => index);
  const attentionMask = Array.from({ length: sequenceLength }, () => 1);

  return {
    input_ids: createInt64Tensor(tokenIds, dims),
    attention_mask: createInt64Tensor(attentionMask, dims),
    position_ids: createInt64Tensor(positions, dims),
  };
}

function getLogitShape(logits: Tensor): [number, number, number] {
  const dims = logits.dims;
  if (dims.length !== 3) {
    throw new Error(`Expected 3D logits, received shape [${dims.join(", ")}].`);
  }
  return [dims[0], dims[1], dims[2]];
}

function getLogitRow(logits: Tensor, position: number): Float32Array {
  const [, sequenceLength, vocabSize] = getLogitShape(logits);
  if (position < 0 || position >= sequenceLength) {
    throw new Error(`Logit position ${position} is outside sequence length ${sequenceLength}.`);
  }
  const start = position * vocabSize;
  const data = logits.data as Float32Array;
  return data.subarray(start, start + vocabSize);
}

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

async function runModel(tokenIds: number[]): Promise<Tensor> {
  if (!model) throw new Error("Model has not loaded.");
  if (tokenIds.length === 0) throw new Error("Cannot run inference with an empty token list.");
  const outputs = await model(createModelInputs(tokenIds));
  return outputs.logits as Tensor;
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

    if (message.type === "getTopNextTokens") {
      const logits = await runModel(message.tokenIds);
      const row = getLogitRow(logits, message.tokenIds.length - 1);
      postResult(message.id, topTokensFromLogits(row, message.n, decodeTokenSync));
      return;
    }

    if (message.type === "getPromptTokenProbabilities") {
      if (message.tokenIds.length === 0) {
        postResult(message.id, []);
        return;
      }
      if (message.tokenIds.length === 1) {
        postResult(message.id, [0.5]);
        return;
      }

      const logits = await runModel(message.tokenIds);
      const probabilities = message.tokenIds.map((tokenId, index) => {
        if (index === 0) return 0.5;
        return probabilityForToken(getLogitRow(logits, index - 1), tokenId);
      });
      postResult(message.id, probabilities);
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
