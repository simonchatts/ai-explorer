# AI Explorer

AI Explorer is web app for visualising how an AI language model chooses the next token. Try it at [explorer.chatts.net](https://explorer.chatts.net) - Chrome is the most likely browser to work right now, because it uses WebGPU to run the model directly in the browser.

This project is inspired by [willkurt/token-explorer](https://github.com/willkurt/token-explorer). 

![Screenshot of AI Explorer](screenshot.png)

## Keyboard Shortcuts

- `ArrowRight` or `Enter`: sample the next token from the visible candidates.
- `ArrowLeft` or `Backspace`: delete one generated token.
- `Space`: continue sampling tokens.
- `Space` or `Escape`: stop continuing while generation is running.

Shortcuts are ignored while you are typing in the prompt or using a control.

## Development

After cloning the repo, first run `npm install`, and then run locally using

    npm run build && (cd dist; python -m http.server)

and viewing at [localhost:8000](http://localhost:8000).
