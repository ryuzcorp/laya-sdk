# laya-sdk

Effect-first TypeScript SDK for **local Laya decision models** (ONNX).
Same `Decision` shapes as a cloud provider, answered on-device — browser or
Node. No API keys, no egress.

```sh
npm install laya-sdk effect   # effect is a peer dependency (v4)
```

```ts
import { loadLayaRuntime, makeLayaDecisionLive } from "laya-sdk";
import { Decision, DecisionModel } from "effect/unstable/ai";
import { Effect, Schema } from "effect";

const rt = await loadLayaRuntime(); // zip (~434MB) → unzip → cache (IDB or ~/.laya)

const definition = Decision.make({
  input: Schema.String,
  decisions: {
    urgent: Decision.probability({
      instructions: "Is this an urgent production blocker?",
      criteria: { false: "No time pressure", true: "Needs action now" },
    }),
  },
});

const { answers } = await Effect.runPromise(
  DecisionModel.decide(definition, {
    input: "The database disk is 100% full and writes are failing.",
  }).pipe(Effect.provide(makeLayaDecisionLive(rt))),
);
// => { urgent: { probability: 0.93… } }
```

## Options

```ts
await loadLayaRuntime({
  modelUrl: "https://my-cdn/laya_int8.onnx", // default: zipped HF int8 release
  tokenizerRepo: "Mattepiu/laya-onnx", // default
  numThreads: 1, // default (no COOP/COEP needed)
  executionProviders: ["wasm"], // default
  onProgress: (label, loaded, total) => {}, // download progress
  cacheDir: "~/.laya", // Node only: default ~/.laya
});
```

`.zip` URLs are unzipped transparently (one `.onnx` entry expected);
`.gz` URLs are gunzipped (see Transfer below). Decoded weights are cached
keyed by URL — IndexedDB in browsers, `~/.laya` files in Node
(`LAYA_CACHE_DIR` env or `cacheDir` option overrides) — repeat loads skip
the network entirely. `clearCachedModel()` drops the cached weights
(force-refresh a re-released model at the same URL).

## Model (measured 2026-09-20)

int8 weights + fp32 embeddings: **434 MB zip** transfer, 554 MB decoded. ✅ verified.

Rules: **marker dim is fixed at 2** (binary choice and noul only),
512-token budget, English text.

## Transfer (no accuracy impact — unzip/gunzip are byte-identical)

Default is `laya_int8.zip` (434 MB, hosted on HF with CORS + `content-length`).
Self-hosting instead? Either zip the `.onnx` (one entry) or
`gzip -9 -k laya_int8.onnx` (554 MB -> 434 MB, measured) and serve with CORS

- `content-length` (progress bar), **without** `content-encoding` — the SDK
  decodes from the file suffix itself. (If your host auto-gzips with
  `content-encoding: gzip`, `fetch` already decoded it and the SDK skips
  re-decoding.) Brotli is skipped: marginal extra saving, spottier browser
  support.

Quantization stays on the shelf until an eval harness can prove zero
accuracy loss — the transfer win above costs nothing in quality.

## Requirements

- `effect` ^4.0.0-rc.112 (peer), modern browser or Node 22+ / Bun.
- First load downloads + unzips weights into a cache (IndexedDB in
  browsers, `~/.laya` in Node); repeat loads read from disk (no network).
  Pass a Blob/object URL as `modelUrl` to bypass download entirely —
  `fetch` handles it.

## Acknowledgements

- [Laya](https://laya.convaiinnovations.com/) — the model this SDK runs on-device.
- [Mattepiu/laya-onnx](https://huggingface.co/Mattepiu/laya-onnx) — the ONNX
  weights and tokenizer this SDK loads.

## License

MIT
