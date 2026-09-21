import { head } from "@ilha/router";
import { atom, watch } from "ilha";
import { Effect, Schema } from "effect";
import { Decision, DecisionModel } from "effect/unstable/ai";
import { makeLayaDecisionLive, type LayaRuntime } from "laya-sdk";
import { createLayaWorkerRuntime, type LayaWorkerHandle } from "../layaWorkerClient";

type Qtype = "classify" | "probability" | "rate";

let rt: LayaRuntime | undefined;

export default function Laya() {
  head({ title: "Laya" });

  // Shareable URLs, classifier.dev style: ?labels=a,b&text=... (+ aliases
  // input/q/state for text, classes/categories/options for labels).
  const qp = new URLSearchParams(location.search);
  const qpGet = (...keys: string[]) => {
    for (const k of keys) {
      const v = qp.get(k);
      if (v !== null && v !== "") return v;
    }
    return undefined;
  };
  const qpType = qpGet("type", "qtype");
  const qpLabels = qpGet("labels", "classes", "categories", "options", "levels")
    ?.split(/[\n,]/)
    .map((o) => o.trim())
    .filter(Boolean)
    .join("\n");
  const autoRun = qp.get("run") === "1" || qp.get("autorun") === "1";

  const status = atom("idle");
  const progress = atom("");
  const qtype = atom<Qtype>(qpType === "probability" || qpType === "rate" ? qpType : "classify");
  const instructions = atom(qpGet("instructions") ?? "Which team should handle this?");
  const optionsText = atom(qpLabels ?? "billing\ntechnical");
  const falseDesc = atom(qpGet("false") ?? "No time pressure");
  const trueDesc = atom(qpGet("true") ?? "Needs action now");
  const state = atom(
    qpGet("text", "input", "q", "state") ?? "My card was charged twice, please fix this today",
  );
  const answer = atom("");
  const latency = atom("");
  const msg = atom("");
  const busy = atom(false);

  let workerHandle: LayaWorkerHandle | undefined;

  const load = async () => {
    busy.set(true);
    msg.set("");
    status.set("loading");
    try {
      // Heavy work (download, unzip, ORT session, inference) runs in a
      // DedicatedWorker; score() calls cross the thread boundary.
      workerHandle = createLayaWorkerRuntime((label, loaded, total) => {
        progress.set(
          label === "weights" && loaded !== undefined
            ? `weights ${(loaded / 1048576).toFixed(0)}MB${total ? ` / ${(total / 1048576).toFixed(0)}MB` : ""}`
            : label,
        );
      });
      await workerHandle.load();
      rt = workerHandle.runtime;
      status.set("ready");
      progress.set("");
      if (autoRun) void run();
    } catch (e) {
      status.set("error");
      msg.set(e instanceof Error ? e.message : "Model failed to load");
    } finally {
      busy.set(false);
    }
  };

  // Keep the URL shareable: every run writes the current form back to ?…
  const syncUrl = (options: string[]) => {
    const out = new URLSearchParams();
    out.set("type", qtype());
    out.set("instructions", instructions());
    out.set("labels", options.join(","));
    if (qtype() === "probability") {
      out.set("false", falseDesc());
      out.set("true", trueDesc());
    }
    out.set("text", state());
    history.replaceState(null, "", `?${out}`);
  };

  const share = async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      msg.set("Link copied");
    } catch {
      msg.set("Copy failed — copy the URL manually");
    }
  };

  const run = async () => {
    if (!rt) {
      msg.set("Load the int8 model first");
      return;
    }
    const options = optionsText()
      .split("\n")
      .map((o) => o.trim())
      .filter(Boolean);
    if (qtype() !== "probability" && options.length !== 2) {
      msg.set("Classify and rate take exactly two options (int8 marker dim is 2)");
      return;
    }
    busy.set(true);
    msg.set("");
    answer.set("");
    syncUrl(options);
    try {
      const def =
        qtype() === "classify"
          ? Decision.make({
              input: Schema.String,
              decisions: {
                pick: Decision.classify({
                  instructions: instructions(),
                  criteria: Object.fromEntries(options.map((o) => [o, o])),
                }),
              },
            })
          : qtype() === "probability"
            ? Decision.make({
                input: Schema.String,
                decisions: {
                  urgent: Decision.probability({
                    instructions: instructions(),
                    criteria: { false: falseDesc(), true: trueDesc() },
                  }),
                },
              })
            : Decision.make({
                input: Schema.String,
                decisions: {
                  level: Decision.rate({
                    instructions: instructions(),
                    criteria: options,
                  }),
                },
              });
      const t0 = performance.now();
      const res = await Effect.runPromise(
        DecisionModel.decide(def, { input: state() }).pipe(
          Effect.provide(makeLayaDecisionLive(rt)),
        ),
      );
      latency.set(`${Math.round(performance.now() - t0)}ms local`);
      answer.set(JSON.stringify(res.answers, null, 2));
    } catch (e) {
      msg.set(e instanceof Error ? e.message : "Inference failed");
    } finally {
      busy.set(false);
    }
  };

  // Mount-only work runs in watch.once: writes rerun the component, so
  // calling load() in the body re-triggers itself forever.
  watch.once(() => {
    void load();
    return () => workerHandle?.dispose();
  });

  return (
    <div class="mx-auto mt-8 flex max-w-xl flex-col gap-4 px-4">
      <div class="flex items-baseline justify-between">
        <h1 class="text-2xl font-bold">Laya test view</h1>
        <a
          class="link text-sm opacity-70"
          href="https://github.com/ryuzcorp/laya-sdk"
          target="_blank"
          rel="noopener"
        >
          Source
        </a>
      </div>
      <div class="card bg-base-100 shadow">
        <div class="card-body gap-2">
          <p class="text-sm opacity-70">
            <code>Mattepiu/laya-onnx</code> · int8 ONNX · runs fully in your browser via
            transformers.js tokenizer + onnxruntime-web.
          </p>
          <p class="text-sm">
            Status: <b>{status()}</b> {progress() ? <span>· {progress()}</span> : ""}
          </p>
        </div>
      </div>
      {msg() ? <div class="alert alert-error text-sm">{msg()}</div> : ""}
      <div class="card bg-base-100 shadow">
        <div class="card-body gap-3">
          <h2 class="font-bold">Decision</h2>
          <label class="form-control w-full">
            <span class="label label-text py-1">Question type</span>
            <select
              class="select select-bordered select-sm w-full"
              value={qtype()}
              onchange={(e) => qtype.set((e.target as HTMLSelectElement).value as Qtype)}
            >
              <option value="classify">classify (choice)</option>
              <option value="probability">probability (noul)</option>
              <option value="rate">rate (score)</option>
            </select>
          </label>
          <label class="form-control w-full">
            <span class="label label-text py-1">Instructions</span>
            <input
              class="input input-bordered input-sm w-full"
              placeholder="Instructions"
              value={instructions()}
              oninput={(e) => instructions.set((e.target as HTMLInputElement).value)}
            />
          </label>
          {qtype() === "probability" ? (
            <div class="flex gap-2">
              <label class="form-control flex-1">
                <span class="label label-text py-1">False means</span>
                <input
                  class="input input-bordered input-sm w-full"
                  placeholder="false means…"
                  value={falseDesc()}
                  oninput={(e) => falseDesc.set((e.target as HTMLInputElement).value)}
                />
              </label>
              <label class="form-control flex-1">
                <span class="label label-text py-1">True means</span>
                <input
                  class="input input-bordered input-sm w-full"
                  placeholder="true means…"
                  value={trueDesc()}
                  oninput={(e) => trueDesc.set((e.target as HTMLInputElement).value)}
                />
              </label>
            </div>
          ) : (
            <label class="form-control w-full">
              <span class="label label-text py-1">Options, one per line</span>
              <textarea
                class="textarea textarea-bordered w-full text-sm"
                rows={3}
                placeholder="Exactly two options (classify) or levels (rate), one per line"
                oninput={(e) => optionsText.set((e.target as HTMLTextAreaElement).value)}
              >
                {optionsText()}
              </textarea>
            </label>
          )}
          <label class="form-control w-full">
            <span class="label label-text py-1">Text to evaluate</span>
            <textarea
              class="textarea textarea-bordered w-full text-sm"
              rows={3}
              placeholder="State to evaluate"
              oninput={(e) => state.set((e.target as HTMLTextAreaElement).value)}
            >
              {state()}
            </textarea>
          </label>
          <div class="flex gap-2">
            <button
              class="btn btn-primary btn-sm w-fit"
              disabled={busy() || status() !== "ready"}
              onclick={() => void run()}
            >
              Classify locally
            </button>
            <button class="btn btn-ghost btn-sm w-fit" onclick={() => void share()}>
              Copy link
            </button>
          </div>
        </div>
      </div>
      {answer() ? (
        <div class="card bg-base-100 shadow">
          <div class="card-body gap-2">
            <h2 class="font-bold">
              Answer <span class="text-sm font-normal opacity-60">· {latency()}</span>
            </h2>
            <pre class="overflow-x-auto rounded bg-base-200 p-2 text-xs">{answer()}</pre>
          </div>
        </div>
      ) : (
        ""
      )}
      <p class="text-xs opacity-50">
        Weights: https://huggingface.co/buckets/ryuz/laya/resolve/laya_int8.zip ·{" "}
        <a class="link" href="https://github.com/ryuzcorp/laya-sdk" target="_blank" rel="noopener">
          Source
        </a>
      </p>
    </div>
  );
}
