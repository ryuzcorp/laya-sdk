import type { InferenceSession, Tensor } from "onnxruntime-web";
import {
  buildLayaInputs,
  extractOnnxFromZip,
  resolveSpecialIds,
  softmax,
  type QtypeWord,
} from "./pure.js";
import { LAYA_REPO, resolveModelUrl, type LayaOptions } from "./models.js";
import { clearCachedModel, getCachedModel, putCachedModel } from "./cache.js";

type OrtNS = typeof import("onnxruntime-web");

export interface LayaEncoder {
  encode(text: string): number[];
  maskId: number;
  clsId: number;
  sepId: number;
}

export interface LayaRuntime {
  encoder: LayaEncoder;
  score(
    qtype: 0 | 1 | 2,
    qtypeWord: QtypeWord,
    instructions: string,
    options: string[],
    state: string,
  ): Promise<number[]>;
}

function toFeeds(
  ort: OrtNS,
  inputIds: number[],
  markers: number[],
  qtype: number,
): Record<string, Tensor> {
  const big = (xs: number[]) => new BigInt64Array(xs.map((x) => BigInt(x)));
  return {
    input_ids: new ort.Tensor("int64", big(inputIds), [1, inputIds.length]),
    attention_mask: new ort.Tensor("int64", new BigInt64Array(inputIds.length).fill(1n), [
      1,
      inputIds.length,
    ]),
    marker_pos: new ort.Tensor("int64", big(markers), [1, markers.length]),
    marker_mask: new ort.Tensor("bool", new Uint8Array(markers.length).fill(1), [
      1,
      markers.length,
    ]),
    qtype: new ort.Tensor("int64", new BigInt64Array([BigInt(qtype)]), [1]),
  };
}

// Transfer compression is byte-identical after decoding, so accuracy is
// untouched. Auto-detected by `.gz` suffix; the server must send the file
// WITHOUT content-encoding (if fetch already decoded it, skip re-decoding).
async function maybeGunzip(
  url: string,
  buf: Uint8Array,
  res: Response,
  onProgress?: (label: string, loaded?: number, total?: number) => void,
): Promise<Uint8Array> {
  const encoded = res.headers.get("content-encoding") ?? "";
  const wantsGzip = url.split("?")[0].toLowerCase().endsWith(".gz") && !/gzip/i.test(encoded);
  if (!wantsGzip) return buf;
  onProgress?.("decompress");
  const raw = await new Response(
    new Blob([buf.buffer as ArrayBuffer]).stream().pipeThrough(new DecompressionStream("gzip")),
  ).arrayBuffer();
  return new Uint8Array(raw);
}

let cached: { optionsKey: string; promise: Promise<LayaRuntime> } | undefined;

export function loadLayaRuntime(options: LayaOptions = {}): Promise<LayaRuntime> {
  const key = JSON.stringify({
    u: resolveModelUrl(options),
    t: options.tokenizerRepo ?? LAYA_REPO,
    m: options.markers ?? 2,
    c: options.cacheDir ?? null,
    n: options.numThreads ?? 1,
    e: options.executionProviders ?? ["wasm"],
  });
  if (cached?.optionsKey !== key) {
    cached = { optionsKey: key, promise: init(options) };
  }
  return cached.promise;
}

async function init(options: LayaOptions): Promise<LayaRuntime> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const onProgress = options.onProgress;
  const expectedMarkers = options.markers ?? 2;
  const [ort, hf] = await Promise.all([
    import("onnxruntime-web"),
    import("@huggingface/transformers"),
  ]);
  // ort resolves its own .wasm/.mjs relatively (node_modules in dev,
  // traced assets in build) — single thread, no special headers needed.
  // Must precede session creation to take effect.
  ort.env.wasm.numThreads = options.numThreads ?? 1;
  onProgress?.("tokenizer");
  const tz = await hf.AutoTokenizer.from_pretrained(options.tokenizerRepo ?? LAYA_REPO, {
    progress_callback: (p: { status?: string; file?: string }) => {
      if (p?.file) onProgress?.(`tokenizer: ${p.file}`);
    },
  });
  const encoder: LayaEncoder = {
    encode: (text: string) => tz.encode(text, { add_special_tokens: false }) as number[],
    ...resolveSpecialIds(tz),
  };
  const modelUrl = resolveModelUrl(options);
  const download = async (): Promise<Uint8Array> => {
    onProgress?.("weights", 0, undefined);
    const res = await fetchImpl(modelUrl);
    if (!res.ok || !res.body) {
      throw new Error(`laya: model download failed (${res.status})`);
    }
    const total = Number(res.headers.get("content-length")) || undefined;
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress?.("weights", loaded, total);
    }
    const buf = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) {
      buf.set(c, off);
      off += c.length;
    }
    chunks.length = 0;
    if (modelUrl.split("?")[0].toLowerCase().endsWith(".zip")) {
      onProgress?.("unzip", loaded, undefined);
      return extractOnnxFromZip(buf);
    }
    return maybeGunzip(modelUrl, buf, res, onProgress);
  };
  const hit = await getCachedModel(modelUrl, options.cacheDir);
  if (hit) onProgress?.("cache", hit.bytes.length, hit.bytes.length);
  let modelBytes: Uint8Array =
    hit?.bytes ??
    (await download().then(async (b) => {
      await putCachedModel(modelUrl, b, options.cacheDir);
      return b;
    }));
  const createSession = () =>
    ort.InferenceSession.create(modelBytes.buffer as ArrayBuffer, {
      executionProviders: options.executionProviders ?? ["wasm"],
    });
  let session: InferenceSession;
  try {
    session = await createSession();
  } catch (e) {
    if (!hit) throw e;
    await clearCachedModel(modelUrl, options.cacheDir);
    modelBytes = await download().then(async (b) => {
      await putCachedModel(modelUrl, b, options.cacheDir);
      return b;
    });
    session = await createSession();
  }
  onProgress?.("session");
  const outputName = session.outputNames[0];
  return {
    encoder,
    score: async (qtype, qtypeWord, instructions, options, state) => {
      const { inputIds, markers } = buildLayaInputs(encoder, {
        qtypeWord,
        instructions,
        options,
        state,
      });
      // Known exports fix marker dim at 2. Binary choice and noul only.
      if (markers.length !== expectedMarkers) {
        throw new Error(
          `laya: this export takes exactly ${expectedMarkers} options (got ${markers.length}) — use binary choice or noul`,
        );
      }
      const out = await session.run(toFeeds(ort, inputIds, markers, qtype));
      const data = out[outputName]?.data as ArrayLike<number> | undefined;
      const logits = (data ? Array.from(data) : []).slice(0, markers.length).map(Number);
      if (logits.length !== markers.length) {
        throw new Error(`laya: expected ${markers.length} logits, got ${logits.length}`);
      }
      return softmax(logits);
    },
  };
}
