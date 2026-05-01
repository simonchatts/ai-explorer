# Web Token Explorer

Web Token Explorer is a browser-based next-token workbench for exploring how a local language model continues text. It runs inference fully in the browser with WebGPU.

The app uses the browser-ready ONNX version of `Qwen/Qwen2.5-0.5B` (`onnx-community/Qwen2.5-0.5B`).

## What You Can Do

- Edit a base prompt.
- Inspect the current prompt and generated completion.
- View the top next-token candidates and probabilities.
- Click an exact token to append it.
- Use **Next** to sample one token from the visible candidate list.
- Use **Continue** to repeatedly sample until cancelled or EOS.
- Delete generated tokens back to the original base prompt.
- Toggle token IDs instead of token text.
- Toggle probability-colored token chips for the full prompt.

## Requirements

- A desktop browser with WebGPU support, such as current Chrome or Edge.
- Enough disk and memory for a local model download.
- Node.js and npm for local development.

The first browser load downloads model files through Transformers.js. Later visits should be faster when browser caching is available.

## Getting Started

Install dependencies:

```sh
npm install
```

Start the local dev server:

```sh
npm run dev
```

Then open the URL printed by Vite, usually:

```text
http://localhost:5173/
```

Build the production bundle:

```sh
npm run build
```

Run unit tests:

```sh
npm run test
```

## Loading Behavior

On desktop-like browsers, the app starts loading the model immediately and shows progress.

On mobile, tablet, narrow, or low-memory-looking devices, the app asks before starting the model download. This is only a user experience heuristic, not a capability guarantee.

If WebGPU is unavailable or model loading fails, the app shows a recoverable error with a retry action. There is no CPU fallback in this version.

## Usage Notes

The base prompt is editable only while no generated completion tokens exist. Once you append or sample a token, the prompt locks. Use **Delete** or **Delete All** to remove generated tokens back to the base prompt and unlock editing.

**Next** and **Continue** intentionally sample only from the visible top candidate list, matching the Python GUI behavior. They do not sample from the model's full vocabulary distribution.

Probability coloring is informational only; each chip also shows a numeric percentage so color is not the only signal.
