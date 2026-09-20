// Pure helpers: token math, input building, answer mapping, zip extract.
import { unzipSync } from "fflate";

export type QtypeWord = "choice" | "score" | "noul";

// Pull the single .onnx payload out of a zipped release (sync, pure).
export function extractOnnxFromZip(zipped: Uint8Array): Uint8Array {
  const files = unzipSync(zipped);
  const names = Object.keys(files).filter((n) => n.toLowerCase().endsWith(".onnx"));
  if (names.length !== 1) {
    throw new Error(`laya: expected exactly one .onnx in zip, found ${names.length}`);
  }
  return files[names[0]!] as Uint8Array;
}

export function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const ex = logits.map((x) => Math.exp(x - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / s);
}

// Pure input builder mirroring the Laya card:
// [CLS] <qtype> question: <instructions> [SEP] ([MASK] <opt>)* [SEP] <state> [SEP]
export function buildLayaInputs(
  enc: {
    encode(text: string): number[];
    maskId: number;
    clsId: number;
    sepId: number;
  },
  args: {
    qtypeWord: QtypeWord;
    instructions: string;
    options: string[];
    state: string;
  },
): { inputIds: number[]; markers: number[] } {
  const ids = [
    enc.clsId,
    ...enc.encode(`${args.qtypeWord} question: ${args.instructions}`),
    enc.sepId,
  ];
  const markers: number[] = [];
  for (const opt of args.options) {
    ids.push(enc.maskId);
    markers.push(ids.length - 1);
    ids.push(...enc.encode(` ${opt}`));
  }
  ids.push(enc.sepId);
  // card: state capped at 256 tokens, hard budget of 512 total
  const room = Math.max(0, 512 - ids.length - 1);
  const stateIds = enc.encode(args.state).slice(0, Math.min(256, room));
  return { inputIds: [...ids, ...stateIds, enc.sepId], markers };
}

export const toClassifyAnswer = (labels: string[], probs: number[]) => {
  const bi = probs.indexOf(Math.max(...probs));
  return {
    label: labels[bi],
    probabilities: Object.fromEntries(labels.map((l, i) => [l, probs[i]])),
    confidence: probs[bi],
  };
};

export const toProbabilityAnswer = (p: number) => ({ probability: p });

export const toRateAnswer = (levels: string[], probs: number[]) => {
  const bi = probs.indexOf(Math.max(...probs));
  return {
    rating: probs.reduce((acc, p, i) => acc + p * i, 0),
    label: levels[bi],
    probabilities: Object.fromEntries(levels.map((l, i) => [l, probs[i]])),
    confidence: probs[bi],
  };
};

// Tokenizer special-id resolution: props → encode/decode round-trip → loud
// failure with diagnostics (never silent wrong ids: inference would be garbage).
export function resolveSpecialIds(tz: {
  mask_token_id?: unknown;
  cls_token_id?: unknown;
  sep_token_id?: unknown;
  encode?: unknown;
  decode?: unknown;
}): { maskId: number; clsId: number; sepId: number } {
  const fromProp = (v: unknown): number | undefined => (typeof v === "number" ? v : undefined);
  const fromEncode = (token: string): number | undefined => {
    if (typeof tz.encode !== "function") return undefined;
    try {
      const ids = (tz.encode as (text: string, opts: Record<string, unknown>) => unknown)(token, {
        add_special_tokens: false,
      });
      if (!Array.isArray(ids) || ids.length !== 1 || typeof ids[0] !== "number") {
        return undefined;
      }
      if (typeof tz.decode === "function") {
        const back = (tz.decode as (ids: number[]) => unknown)([ids[0]]);
        if (typeof back !== "string" || !back.includes(token)) return undefined;
      }
      return ids[0];
    } catch {
      return undefined;
    }
  };
  const maskId = fromProp(tz.mask_token_id) ?? fromEncode("[MASK]");
  const clsId = fromProp(tz.cls_token_id) ?? fromEncode("[CLS]");
  const sepId = fromProp(tz.sep_token_id) ?? fromEncode("[SEP]");
  if (maskId === undefined || clsId === undefined || sepId === undefined) {
    throw new Error(
      `laya: tokenizer is missing CLS/SEP/MASK ids ` +
        `(cls=${String(clsId)} sep=${String(sepId)} mask=${String(maskId)}, ` +
        `hasEncode=${typeof tz.encode})`,
    );
  }
  if (new Set([maskId, clsId, sepId]).size !== 3) {
    throw new Error(
      `laya: tokenizer ids collapsed (cls=${clsId} sep=${sepId} mask=${maskId}) — refusing silent garbage`,
    );
  }
  return { maskId, clsId, sepId };
}
