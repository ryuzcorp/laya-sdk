import { head } from "@ilha/router";
import { atom, watch } from "ilha";
import { Effect, Schema } from "effect";
import { Decision, DecisionModel } from "effect/unstable/ai";
import {
  LAYA_INT8_ZIP_URL,
  LAYA_REPO,
  makeLayaDecisionLive,
  type LayaRuntime,
} from "laya-sdk";
import { createLayaWorkerRuntime, type LayaWorkerHandle } from "../layaWorkerClient";

type Qtype = "classify" | "probability" | "rate";

let rt: LayaRuntime | undefined;

export default function Laya() {
  head({ title: "Laya" });

  const status = atom("idle");
  const progress = atom("");
  const qtype = atom<Qtype>("classify");
  const instructions = atom("Which team should handle this?");
  const optionsText = atom("billing\ntechnical");
  const falseDesc = atom("No time pressure");
  const trueDesc = atom("Needs action now");
  const state = atom("My card was charged twice, please fix this today");
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
            : label
        );
      });
      await workerHandle.load();
      rt = workerHandle.runtime;
      status.set("ready");
      progress.set("");
    } catch (e) {
      status.set("error");
      msg.set(e instanceof Error ? e.message : "Model failed to load");
    } finally {
      busy.set(false);
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
          Effect.provide(makeLayaDecisionLive(rt))
        )
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
      <h1 class="text-2xl font-bold">Laya test view</h1>
      <div class="card bg-base-100 shadow">
        <div class="card-body gap-2">
          <p class="text-sm opacity-70">
            <code>{LAYA_REPO}</code> · int8 ONNX · runs fully in your browser
            via transformers.js tokenizer + onnxruntime-web.
          </p>
          <p class="text-sm">
            Status: <b>{status()}</b> {progress() ? <span>· {progress()}</span> : ""}
          </p>
        </div>
      </div>
      {msg() ? <div class="alert alert-error text-sm">{msg()}</div> : ""}
      <div class="card bg-base-100 shadow">
        <div class="card-body gap-2">
          <h2 class="font-bold">Decision</h2>
          <label class="text-sm opacity-70">
            Question type
            <select
              class="select select-bordered select-sm ml-2"
              value={qtype()}
              onchange={(e) =>
                qtype.set((e.target as HTMLSelectElement).value as Qtype)
              }
            >
              <option value="classify">classify (choice)</option>
              <option value="probability">probability (noul)</option>
              <option value="rate">rate (score)</option>
            </select>
          </label>
          <input
            class="input input-bordered input-sm"
            placeholder="Instructions"
            value={instructions()}
            oninput={(e) => instructions.set((e.target as HTMLInputElement).value)}
          />
          {qtype() === "probability" ? (
            <div class="flex gap-2">
              <input
                class="input input-bordered input-sm"
                placeholder="false means…"
                value={falseDesc()}
                oninput={(e) => falseDesc.set((e.target as HTMLInputElement).value)}
              />
              <input
                class="input input-bordered input-sm"
                placeholder="true means…"
                value={trueDesc()}
                oninput={(e) => trueDesc.set((e.target as HTMLInputElement).value)}
              />
            </div>
          ) : (
            <textarea
              class="textarea textarea-bordered text-sm"
              rows={3}
              placeholder="Exactly two options (classify) or levels (rate), one per line"
              oninput={(e) => optionsText.set((e.target as HTMLTextAreaElement).value)}
            >
              {optionsText()}
            </textarea>
          )}
          <textarea
            class="textarea textarea-bordered text-sm"
            rows={3}
            placeholder="State to evaluate"
            oninput={(e) => state.set((e.target as HTMLTextAreaElement).value)}
          >
            {state()}
          </textarea>
          <button
            class="btn btn-primary btn-sm w-fit"
            disabled={busy() || status() !== "ready"}
            onclick={() => void run()}
          >
            Classify locally
          </button>
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
      <p class="text-xs opacity-50">Weights: {LAYA_INT8_ZIP_URL}</p>
    </div>
  );
}
