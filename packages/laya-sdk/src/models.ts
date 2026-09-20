// Model location + loading options. Default is the zipped int8 release
// (434 MiB transfer, unzipped in-memory, cached decoded: IndexedDB in
// browsers, ~/.laya in Node). Pass `modelUrl` to override: raw `.onnx`
// or `.gz` URLs still work.
export const LAYA_REPO = "Mattepiu/laya-onnx";
export const LAYA_INT8_ZIP_URL =
  "https://huggingface.co/buckets/ryuz/laya/resolve/laya_int8.zip?download=true";

export interface LayaOptions {
  /** Weight file URL (or same-origin path). Defaults to the int8 release. */
  modelUrl?: string;
  /** Exact marker slots this export supports. Defaults to 2. */
  markers?: number;
  /** Tokenizer repo. Defaults to LAYA_REPO. */
  tokenizerRepo?: string;
  /** ORT threads. Defaults to 1 (no COOP/COEP headers needed). */
  numThreads?: number;
  /** ORT execution providers. Defaults to ["wasm"]. */
  executionProviders?: string[];
  /** Fetch implementation (override for tests/proxies). Defaults to global fetch. */
  fetchImpl?: typeof fetch;
  /** Node fs cache dir (default ~/.laya, LAYA_CACHE_DIR wins unless set).
   * Ignored in browsers (IndexedDB). */
  cacheDir?: string;
  /** (label, loadedBytes, totalBytes?) progress callback. */
  onProgress?: (label: string, loaded?: number, total?: number) => void;
}

export function resolveModelUrl(options: LayaOptions = {}): string {
  return options.modelUrl ?? LAYA_INT8_ZIP_URL;
}
