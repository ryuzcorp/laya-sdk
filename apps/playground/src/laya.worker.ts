// Dedicated-worker entry: all heavy Laya work (download, unzip, ORT
// session, inference) stays off the main thread. Loaded via `?worker`.
// Transport uses effect/unstable/workers: WorkerRunnerPlatform here,
// WorkerPlatform + Spawner on the page side (see layaWorkerClient.ts).
import { Effect, Layer } from "effect";
import { WorkerRunner } from "effect/unstable/workers";
import { loadLayaRuntime, type LayaRuntime } from "laya-sdk";
import type { InMsg, OutMsg } from "./layaWorkerProtocol";

// SAFETY: this module only ever runs as a DedicatedWorker entry (loaded via
// `?worker`), where `self` is the DedicatedWorkerGlobalScope with exactly
// this postMessage/addEventListener/removeEventListener surface.
const workerScope = self as unknown as {
  postMessage(message: unknown): void;
  addEventListener(type: "message", listener: (event: MessageEvent) => void): void;
  removeEventListener(type: "message", listener: (event: MessageEvent) => void): void;
};

const BrowserRunnerLive = Layer.effect(
  WorkerRunner.WorkerRunnerPlatform,
  Effect.succeed({
    start: <O, I>() =>
      Effect.sync((): WorkerRunner.WorkerRunner<O, I> => {
        const runner: WorkerRunner.WorkerRunner<OutMsg, InMsg> = {
          run: (handler) =>
            Effect.acquireUseRelease(
              Effect.sync(() => {
                const listener = (event: MessageEvent) => {
                  const data = event.data as readonly [0, InMsg] | readonly [1];
                  if (data[0] !== 0) return;
                  const out = handler(0, data[1]);
                  // Our handler returns void; runPromise covers the
                  // interface's Effect-returning case with no runtime handle.
                  if (Effect.isEffect(out)) void Effect.runPromise(out as Effect.Effect<void>);
                };
                workerScope.addEventListener("message", listener);
                return listener;
              }),
              () => Effect.never,
              (listener) => Effect.sync(() => workerScope.removeEventListener("message", listener)),
            ),
          send: (_portId, message) => Effect.sync(() => workerScope.postMessage([1, message])),
          sendUnsafe: (_portId, message) => workerScope.postMessage([1, message]),
        };
        workerScope.postMessage([0]);
        // SAFETY: start is only ever instantiated as start<OutMsg, InMsg>
        // by main below; the generic signature just satisfies the platform
        // service shape shared with Node/Bun adapters.
        return runner as unknown as WorkerRunner.WorkerRunner<O, I>;
      }),
  }),
);

// Tiny promise-chain mutex: ORT session.run calls must not overlap.
class Mutex {
  private tail: Promise<unknown> = Promise.resolve();
  run<T>(fn: () => Promise<T>): Promise<T> {
    const prev = this.tail;
    let release!: () => void;
    this.tail = new Promise<void>((resolve) => {
      release = resolve;
    });
    return prev.then(fn).finally(release);
  }
}

const mutex = new Mutex();
let loadStarted = false;
let rtP: Promise<LayaRuntime> | undefined;

const replyError = (
  runner: WorkerRunner.WorkerRunner<OutMsg, InMsg>,
  portId: number,
  id: number,
  message: unknown,
): Effect.Effect<void> =>
  runner.send(portId, {
    id,
    kind: "error",
    message: message instanceof Error ? message.message : String(message),
  });

const handleJob = (
  runner: WorkerRunner.WorkerRunner<OutMsg, InMsg>,
  portId: number,
  msg: InMsg,
): Promise<void> =>
  mutex.run(async () => {
    if (msg.kind === "load") {
      loadStarted = true;
      try {
        rtP ??= loadLayaRuntime({
          onProgress: (label, loaded, total) =>
            runner.sendUnsafe(portId, { id: msg.id, kind: "progress", label, loaded, total }),
        });
        await rtP;
        await Effect.runPromise(runner.send(portId, { id: msg.id, kind: "ready" }));
      } catch (e) {
        await Effect.runPromise(replyError(runner, portId, msg.id, e));
      }
      return;
    }
    if (!loadStarted || rtP === undefined) {
      await Effect.runPromise(replyError(runner, portId, msg.id, "laya: model not loaded"));
      return;
    }
    try {
      const rt = await rtP;
      const probs = await rt.score(
        msg.args.qtype,
        msg.args.qtypeWord,
        msg.args.instructions,
        msg.args.options,
        msg.args.state,
      );
      await Effect.runPromise(runner.send(portId, { id: msg.id, kind: "result", probs }));
    } catch (e) {
      await Effect.runPromise(replyError(runner, portId, msg.id, e));
    }
  });

const main = Effect.gen(function* () {
  const platform = yield* WorkerRunner.WorkerRunnerPlatform;
  const runner = yield* platform.start<OutMsg, InMsg>();
  yield* runner.run((portId, msg) => {
    void handleJob(runner, portId, msg).catch(() => {
      // Unreachable: every branch above replies or catches. Last resort
      // keeps one bad job from killing the runner silently.
      runner.sendUnsafe(portId, {
        id: (msg as InMsg).id,
        kind: "error",
        message: "laya: job failed",
      });
    });
  });
});

Effect.runFork(main.pipe(Effect.provide(BrowserRunnerLive)));
