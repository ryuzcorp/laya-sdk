import { pages } from "@ilha/router/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [
    pages(),
    tailwindcss(),
  ],
  resolve: {
    tsconfigPaths: true,
  },
  // onnxruntime manages its own workers/wasm loading; don't prebundle it.
  // Single-threaded inference needs no COOP/COEP headers (same setup the
  // transformers.js team ships: ort resolves its .wasm/.mjs relatively,
  // served from node_modules in dev and traced as assets in build).
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
