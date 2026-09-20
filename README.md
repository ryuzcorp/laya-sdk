# Laya

Local Laya decision models behind the Effect `Decision` API — no API keys, no egress.

- [`packages/laya-sdk`](packages/laya-sdk) — isomorphic SDK (browser + Node).
  ONNX int8 weights, zipped transfer, IndexedDB / `~/.laya` cache.
- [`apps/playground`](apps/playground) — browser test view. Loads the model
  and runs classify / probability / rate decisions locally.

```sh
bun install
bun --filter laya-sdk test   # SDK unit tests
bun --filter playground dev  # playground at /
```
