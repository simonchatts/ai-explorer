export const MODEL_ID = "onnx-community/Qwen2.5-0.5B";
export const DEFAULT_PROMPT = "Once upon a time, there was a";
export const DEFAULT_TOP_N = 30;

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

export interface LoadProgress {
  progress: number | null;
  message: string;
}

export interface TokenPredictions {
  nextTokens: TokenCandidate[];
  tokenProbabilities: number[] | null;
}

export interface InferenceBackend {
  load(): Promise<void>;
  encode(text: string): Promise<number[]>;
  decode(tokenIds: number[]): Promise<string>;
  decodeToken(tokenId: number): Promise<string>;
  getTokenPredictions(tokenIds: number[], n: number, includeProbabilities: boolean): Promise<TokenPredictions>;
  getTopNextTokens(tokenIds: number[], n: number): Promise<TokenCandidate[]>;
  getPromptTokenProbabilities(tokenIds: number[]): Promise<number[]>;
  getEosTokenId(): Promise<number | null>;
}
