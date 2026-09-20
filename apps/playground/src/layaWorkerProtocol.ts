// Job protocol between the page and laya.worker.ts.
// Plain structured-cloneable data only — no functions, no class instances.
export interface ScoreArgs {
  qtype: 0 | 1 | 2;
  qtypeWord: "choice" | "score" | "noul";
  instructions: string;
  options: string[];
  state: string;
}

export type InMsg = { id: number; kind: "load" } | { id: number; kind: "score"; args: ScoreArgs };

export type OutMsg =
  | { id: number; kind: "progress"; label: string; loaded?: number; total?: number }
  | { id: number; kind: "ready" }
  | { id: number; kind: "result"; probs: number[] }
  | { id: number; kind: "error"; message: string };
