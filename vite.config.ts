import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  worker: {
    // Firefox can parse nested runtime workers as classic scripts; avoid import.meta in the worker bundle.
    format: "iife",
  },
});
