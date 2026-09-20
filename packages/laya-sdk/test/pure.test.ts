import { describe, expect, test } from "bun:test";
import { zipSync } from "fflate";
import {
  buildLayaInputs,
  extractOnnxFromZip,
  resolveSpecialIds,
  softmax,
  toClassifyAnswer,
  toProbabilityAnswer,
  toRateAnswer,
} from "../src/pure.js";
import { LAYA_INT8_ZIP_URL } from "../src/models.js";
import { LAYA_REPO, resolveModelUrl } from "../src/models";

const approx = (a: number[], b: number[]) =>
  a.length === b.length && a.every((x, i) => Math.abs(x - b[i]!) < 1e-9);

describe("softmax", () => {
  test("uniform", () => {
    expect(approx(softmax([0, 0]), [0.5, 0.5])).toBe(true);
  });
  test("sums to 1, argmax last", () => {
    const p = softmax([1, 2, 3]);
    expect(Math.abs(p.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-9);
    expect(p[2]! > p[1]!).toBe(true);
  });
});

describe("buildLayaInputs", () => {
  // stub encoder: ids [CLS]=1 [SEP]=2 [MASK]=3
  const stub = {
    encode: (t: string) => [...t].map((c) => c.charCodeAt(0)),
    maskId: 3,
    clsId: 1,
    sepId: 2,
  };
  const { inputIds, markers } = buildLayaInputs(stub, {
    qtypeWord: "noul",
    instructions: "Urgent?",
    options: ["false", "true"],
    state: "disk full",
  });
  test("starts with CLS", () => expect(inputIds[0]).toBe(1));
  test("ends with SEP", () => expect(inputIds[inputIds.length - 1]).toBe(2));
  test("one marker per option pointing at MASK", () => {
    expect(markers.length).toBe(2);
    expect(markers.every((m) => inputIds[m] === 3)).toBe(true);
  });
  test("markers ascend", () => expect(markers[0]! < markers[1]!).toBe(true));
  test("budget respected", () => expect(inputIds.length <= 512).toBe(true));
});

describe("answer mapping", () => {
  test("classify argmax + probs", () => {
    const c = toClassifyAnswer(["a", "b"], [0.25, 0.75]);
    expect(c.label).toBe("b");
    expect(c.probabilities["b"]).toBe(0.75);
    expect(c.confidence).toBe(0.75);
  });
  test("probability passthrough", () => {
    expect(toProbabilityAnswer(0.8).probability).toBe(0.8);
  });
  test("rate expected value + label", () => {
    const r = toRateAnswer(["low", "high"], [0.2, 0.8]);
    expect(r.label).toBe("high");
    expect(Math.abs(r.rating - 0.8)).toBeLessThan(1e-9);
  });
});

describe("resolveSpecialIds", () => {
  test("ids from props", () => {
    const ids = resolveSpecialIds({
      mask_token_id: 50284,
      cls_token_id: 50281,
      sep_token_id: 50282,
      convert_tokens_to_ids: () => {
        throw new Error("unused");
      },
    });
    expect(ids).toEqual({ maskId: 50284, clsId: 50281, sepId: 50282 });
  });
  test("ids from encoder lookup", () => {
    const vocab: Record<string, number> = {
      "[MASK]": 50284,
      "[CLS]": 50281,
      "[SEP]": 50282,
    };
    const ids = resolveSpecialIds({
      encode: (t: string) => [vocab[t]!],
      decode: (ids: number[]) => Object.keys(vocab).find((k) => vocab[k] === ids[0]) ?? "",
    });
    expect(ids).toEqual({ maskId: 50284, clsId: 50281, sepId: 50282 });
  });
  test("loud failure with diagnostics", () => {
    expect(() => resolveSpecialIds({})).toThrow(/missing CLS\/SEP\/MASK/);
  });
  test("mis-resolved ids refused", () => {
    expect(() => resolveSpecialIds({ encode: () => [50280], decode: () => "[UNK]" })).toThrow();
  });
  test("collapsed ids refused", () => {
    expect(() => resolveSpecialIds({ mask_token_id: 5, cls_token_id: 5, sep_token_id: 5 })).toThrow(
      /collapsed/,
    );
  });
});

describe("models", () => {
  test("int8 default URL", () => {
    expect(resolveModelUrl()).toBe(LAYA_INT8_ZIP_URL);
    expect(resolveModelUrl({})).toBe(LAYA_INT8_ZIP_URL);
  });
  test("zip round-trip extracts the onnx payload", () => {
    const payload = new Uint8Array([1, 2, 3, 4]);
    const zipped = zipSync({ "laya_int8.onnx": payload });
    expect(extractOnnxFromZip(zipped)).toEqual(payload);
    expect(() => extractOnnxFromZip(zipSync({ "a.onnx": payload, "b.onnx": payload }))).toThrow(
      /exactly one/,
    );
  });
  test("modelUrl override wins", () => {
    expect(resolveModelUrl({ modelUrl: "https://x/y.onnx" })).toBe("https://x/y.onnx");
  });
  test("tokenizer repo constant", () => {
    expect(LAYA_REPO).toBe("Mattepiu/laya-onnx");
  });
  test("int8 file reachable and plausibly sized", async () => {
    const res = await fetch(LAYA_INT8_ZIP_URL, { method: "HEAD", redirect: "follow" });
    const size = Number(res.headers.get("content-length") ?? 0);
    expect(res.ok).toBe(true);
    expect(size).toBeGreaterThan(100_000_000);
  });
});
