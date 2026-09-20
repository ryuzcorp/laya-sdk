// Persistent model cache keyed by model URL. Browser: IndexedDB
// (localStorage caps at ~5MB). Node: files under ~/.laya (LAYA_CACHE_DIR
// or `cacheDir` override). Every failure mode degrades to network-only:
// no IDB, no fs, quota/private mode, corrupt entries.
//
// No static `node:` imports: the fs backend loads via dynamic import, so
// browser bundlers never see it — and it only runs when indexedDB is absent.
export interface CachedModel {
  bytes: Uint8Array;
  size: number;
  savedAt: number;
}

interface StoredRow {
  bytes: Uint8Array | ArrayBuffer;
  size: number;
  savedAt: number;
}

const DB_NAME = "laya-sdk";
const STORE = "models";

function factory(): IDBFactory | undefined {
  return typeof indexedDB !== "undefined" ? indexedDB : undefined;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const f = factory();
    if (!f) {
      reject(new Error("laya: IndexedDB unavailable"));
      return;
    }
    const req = f.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("laya: IDB open failed"));
  });
}

function req<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error ?? new Error("laya: IDB request failed"));
  });
}

async function withStore<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => Promise<T>): Promise<T> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, mode);
    const result = await fn(tx.objectStore(STORE));
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error("laya: IDB transaction failed"));
      tx.onabort = () => reject(tx.error ?? new Error("laya: IDB transaction aborted"));
    });
    return result;
  } finally {
    db.close();
  }
}

function sane(row: unknown): CachedModel | undefined {
  if (!row || typeof row !== "object") return undefined;
  const r = row as Partial<StoredRow>;
  const bytes = r.bytes instanceof Uint8Array ? r.bytes : r.bytes instanceof ArrayBuffer ? new Uint8Array(r.bytes) : undefined;
  if (!bytes || !r.size || bytes.length !== r.size) return undefined;
  return { bytes, size: r.size, savedAt: r.savedAt ?? 0 };
}

// --- Node filesystem backend ---------------------------------------------

interface NodeProc {
  versions?: Record<string, string>;
  env?: Record<string, string | undefined>;
}

function nodeProc(): NodeProc | undefined {
  const p = (globalThis as { process?: NodeProc }).process;
  return p?.versions?.node ? p : undefined;
}

function isNode(): boolean {
  return nodeProc() !== undefined && typeof indexedDB === "undefined";
}

interface NodeFs {
  mkdir(path: string, opts: { recursive: boolean }): Promise<unknown>;
  readFile(path: string): Promise<Uint8Array>;
  writeFile(path: string, data: Uint8Array): Promise<void>;
  stat(path: string): Promise<{ size: number }>;
  rm(path: string, opts?: { recursive?: boolean; force?: boolean }): Promise<void>;
  readdir(path: string): Promise<string[]>;
}

async function nodeFs(): Promise<NodeFs | undefined> {
  try {
    // Non-literal spec: no static type resolution, no bundler tracing
    // (@vite-ignore keeps it external in browser builds, where it never runs).
    const spec: string = "node:fs/promises";
    return (await import(/* @vite-ignore */ spec)) as NodeFs;
  } catch {
    return undefined;
  }
}

/** Cache dir: explicit option > LAYA_CACHE_DIR > ~/.laya (> ./.laya fallback). */
export function resolveCacheDir(explicit?: string): string {
  const env = nodeProc()?.env;
  const dir = explicit ?? env?.LAYA_CACHE_DIR;
  if (dir) return dir;
  const home = env?.HOME || env?.USERPROFILE;
  return home ? `${home}/.laya` : ".laya";
}

function fileBase(url: string): string {
  return url
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(-160);
}

interface FileMeta {
  url: string;
  size: number;
  savedAt: number;
}

async function nodeGet(url: string, cacheDir?: string): Promise<CachedModel | undefined> {
  try {
    const fs = await nodeFs();
    if (!fs) return undefined;
    const dir = resolveCacheDir(cacheDir);
    const base = fileBase(url);
    const meta: FileMeta = JSON.parse(
      new TextDecoder().decode(await fs.readFile(`${dir}/${base}.json`))
    );
    if (meta.url !== url || !meta.size) return undefined;
    const st = await fs.stat(`${dir}/${base}.bin`);
    if (st.size !== meta.size) return undefined;
    const bytes = await fs.readFile(`${dir}/${base}.bin`);
    if (bytes.length !== meta.size) return undefined;
    // slice: exact-size backing (fs may hand back pooled buffers, and the
    // loader passes bytes.buffer straight to ORT).
    return { bytes: bytes.slice(), size: meta.size, savedAt: meta.savedAt ?? 0 };
  } catch {
    return undefined;
  }
}

async function nodePut(url: string, bytes: Uint8Array, cacheDir?: string): Promise<void> {
  try {
    const fs = await nodeFs();
    if (!fs) return;
    const dir = resolveCacheDir(cacheDir);
    await fs.mkdir(dir, { recursive: true });
    const base = fileBase(url);
    await fs.writeFile(`${dir}/${base}.bin`, bytes);
    const meta: FileMeta = { url, size: bytes.length, savedAt: Date.now() };
    await fs.writeFile(`${dir}/${base}.json`, new TextEncoder().encode(JSON.stringify(meta)));
  } catch {
    // read-only fs etc: session still works, just not cached
  }
}

async function nodeClear(url?: string, cacheDir?: string): Promise<void> {
  try {
    const fs = await nodeFs();
    if (!fs) return;
    const dir = resolveCacheDir(cacheDir);
    if (url) {
      const base = fileBase(url);
      await fs.rm(`${dir}/${base}.bin`, { force: true });
      await fs.rm(`${dir}/${base}.json`, { force: true });
      return;
    }
    const names = await fs.readdir(dir);
    await Promise.all(
      names
        .filter((n) => n.endsWith(".bin") || n.endsWith(".json"))
        .map((n) => fs.rm(`${dir}/${n}`, { force: true }))
    );
  } catch {
    // best effort
  }
}

// --- Public API: backend dispatch -----------------------------------------

export async function getCachedModel(url: string, cacheDir?: string): Promise<CachedModel | undefined> {
  if (isNode()) return nodeGet(url, cacheDir);
  try {
    const row = await withStore("readonly", (s) => req(s.get(url)));
    return sane(row);
  } catch {
    return undefined;
  }
}

export async function putCachedModel(url: string, bytes: Uint8Array, cacheDir?: string): Promise<void> {
  if (isNode()) {
    await nodePut(url, bytes, cacheDir);
    return;
  }
  try {
    await withStore("readwrite", async (s) => {
      await req(s.put({ bytes, size: bytes.length, savedAt: Date.now() }, url));
    });
  } catch {
    // quota/private mode: session still works, just not cached
  }
}

export async function clearCachedModel(url?: string, cacheDir?: string): Promise<void> {
  if (isNode()) {
    await nodeClear(url, cacheDir);
    return;
  }
  try {
    if (url) {
      await withStore("readwrite", async (s) => {
        await req(s.delete(url));
      });
    } else {
      await withStore("readwrite", async (s) => {
        await req(s.clear());
      });
    }
  } catch {
    // best effort
  }
}
