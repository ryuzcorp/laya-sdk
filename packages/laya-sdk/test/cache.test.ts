// Cache tests with a minimal in-memory fake of the tiny IDB surface
// cache.ts uses, since bun has no IndexedDB. Each op defers to a microtask.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { clearCachedModel, getCachedModel, putCachedModel, resolveCacheDir } from "../src/cache.js";

interface TestFs {
  mkdtemp(prefix: string): Promise<string>;
  rm(path: string, opts: { recursive: boolean; force: boolean }): Promise<void>;
  writeFile(path: string, data: Uint8Array | string): Promise<void>;
}

async function testFs(): Promise<TestFs> {
  // Non-literal spec: no static type resolution needed without @types/node.
  const spec: string = "node:fs/promises";
  return (await import(/* @vite-ignore */ spec)) as TestFs;
}

function testEnv(): Record<string, string | undefined> {
  const p = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  if (!p?.env) throw new Error("node env needed for fs cache tests");
  return p.env;
}

function fakeRequest<T>(fn: () => T) {
  const req: Record<string, unknown> = { result: undefined, error: undefined };
  queueMicrotask(() => {
    try {
      req.result = fn();
      (req.onsuccess as () => void)?.();
    } catch (e) {
      req.error = e;
      (req.onerror as () => void)?.();
    }
  });
  return req;
}

function installFakeIDB(store: Map<unknown, unknown>, hooks: { failPut?: boolean } = {}) {
  const fakeFactory = {
    open() {
      const req: Record<string, unknown> = { result: undefined, error: undefined };
      queueMicrotask(() => {
        (req.result as unknown) = {
          close() {},
          transaction() {
            // Same object identity throughout: withStore assigns
            // tx.oncomplete on this exact object after fn() settles.
            const tx: Record<string, unknown> = {
              error: undefined,
              objectStore() {
                return {
                  get: (key: unknown) =>
                    fakeRequest(() => (store.has(key) ? store.get(key) : undefined)),
                  put: (value: unknown, key: unknown) =>
                    fakeRequest(() => {
                      if (hooks.failPut) throw new Error("QuotaExceededError");
                      store.set(key, value);
                    }),
                  delete: (key: unknown) =>
                    fakeRequest(() => {
                      store.delete(key);
                    }),
                  clear: () =>
                    fakeRequest(() => {
                      store.clear();
                    }),
                };
              },
            };
            // Macrotask: real IDB fires complete only after all requests
            // settle, which is always after the current microtask drain.
            setTimeout(() => (tx.oncomplete as () => void)?.(), 0);
            return tx;
          },
        };
        (req.onsuccess as () => void)?.();
      });
      return req;
    },
  };
  (globalThis as Record<string, unknown>).indexedDB = fakeFactory;
}

function uninstallFakeIDB() {
  delete (globalThis as Record<string, unknown>).indexedDB;
}

const u8 = (s: string) => new TextEncoder().encode(s);

let store: Map<unknown, unknown>;

describe("model cache (idb)", () => {
  beforeEach(() => {
    store = new Map();
    installFakeIDB(store);
  });
  afterEach(() => {
    uninstallFakeIDB();
  });
  test("miss on empty store", async () => {
    expect(await getCachedModel("m")).toBeUndefined();
  });
  test("put/get round-trip keeps bytes, size, savedAt", async () => {
    await putCachedModel("m", u8("hello"));
    const hit = await getCachedModel("m");
    expect(hit?.bytes).toEqual(u8("hello"));
    expect(hit?.size).toBe(5);
    expect(hit!.savedAt).toBeGreaterThan(0);
  });
  test("size mismatch and junk rows read as miss", async () => {
    store.set("m", { bytes: u8("hi"), size: 999, savedAt: 0 });
    expect(await getCachedModel("m")).toBeUndefined();
    store.set("m", { nope: true });
    expect(await getCachedModel("m")).toBeUndefined();
  });
  test("clear(url) removes one entry, clear() wipes all", async () => {
    await putCachedModel("a", u8("1"));
    await putCachedModel("b", u8("2"));
    await clearCachedModel("a");
    expect(await getCachedModel("a")).toBeUndefined();
    expect(await getCachedModel("b")).not.toBeUndefined();
    await clearCachedModel();
    expect(await getCachedModel("b")).toBeUndefined();
  });
  // No-IDB miss is covered by the node-fs suite below (bun has no IndexedDB).
  test("put failure never throws (quota/private mode)", async () => {
    uninstallFakeIDB();
    installFakeIDB(store, { failPut: true });
    await putCachedModel("m", u8("x"));
    expect(await getCachedModel("m")).toBeUndefined();
  });
});

describe("model cache (node fs)", () => {
  let dir: string;
  let savedEnv: string | undefined;
  let envHasKey: boolean;

  beforeEach(async () => {
    uninstallFakeIDB(); // force the node backend under bun
    const fs = await testFs();
    const env = testEnv();
    savedEnv = env.LAYA_CACHE_DIR;
    envHasKey = "LAYA_CACHE_DIR" in env;
    delete env.LAYA_CACHE_DIR;
    dir = await fs.mkdtemp(`${env.TMPDIR ?? env.TEMP ?? "/tmp"}/laya-test-`);
  });

  afterEach(async () => {
    const fs = await testFs();
    await fs.rm(dir, { recursive: true, force: true });
    const env = testEnv();
    if (envHasKey) env.LAYA_CACHE_DIR = savedEnv;
    else delete env.LAYA_CACHE_DIR;
  });

  test("miss on empty dir", async () => {
    expect(await getCachedModel("https://x/model.zip", dir)).toBeUndefined();
  });
  test("put/get round-trip keeps bytes, size, savedAt", async () => {
    await putCachedModel("https://x/model.zip", u8("hello"), dir);
    const hit = await getCachedModel("https://x/model.zip", dir);
    expect(hit?.bytes).toEqual(u8("hello"));
    expect(hit?.size).toBe(5);
    expect(hit!.savedAt).toBeGreaterThan(0);
  });
  test("corrupt entries read as miss", async () => {
    const fs = await testFs();
    const meta = `${dir}/https_x_model_zip.json`;
    await putCachedModel("https://x/model.zip", u8("hello"), dir);
    await fs.writeFile(meta, JSON.stringify({ url: "https://x/model.zip", size: 999, savedAt: 0 }));
    expect(await getCachedModel("https://x/model.zip", dir)).toBeUndefined();
    await putCachedModel("https://x/model.zip", u8("hello"), dir);
    await fs.writeFile(meta, JSON.stringify({ url: "https://x/other.zip", size: 5, savedAt: 0 }));
    expect(await getCachedModel("https://x/model.zip", dir)).toBeUndefined();
    await fs.writeFile(meta, "not json{{{");
    expect(await getCachedModel("https://x/model.zip", dir)).toBeUndefined();
  });
  test("clear(url) removes one entry, clear() wipes all", async () => {
    await putCachedModel("https://x/a.zip", u8("1"), dir);
    await putCachedModel("https://x/b.zip", u8("2"), dir);
    await clearCachedModel("https://x/a.zip", dir);
    expect(await getCachedModel("https://x/a.zip", dir)).toBeUndefined();
    expect(await getCachedModel("https://x/b.zip", dir)).not.toBeUndefined();
    await clearCachedModel(undefined, dir);
    expect(await getCachedModel("https://x/b.zip", dir)).toBeUndefined();
  });
  test("LAYA_CACHE_DIR is honored without explicit dir", async () => {
    testEnv().LAYA_CACHE_DIR = dir;
    await putCachedModel("https://x/env.zip", u8("env"));
    expect(await getCachedModel("https://x/env.zip")).not.toBeUndefined();
  });
  test("resolveCacheDir prefers explicit, defaults under .laya", () => {
    expect(resolveCacheDir("/tmp/custom")).toBe("/tmp/custom");
    expect(resolveCacheDir().endsWith(".laya")).toBe(true);
  });
});
