// Main-thread side of the Laya worker: spawns the DedicatedWorker through
// effect/unstable/workers (Spawner + WorkerPlatform transport, scope-owned
// lifetime) and exposes a plain LayaRuntime proxy — the decisions layer and
// the page keep working unchanged, only score() crosses the thread boundary.
import { Deferred, Effect, Exit, Fiber, Layer, Scope } from "effect";
import { Worker, WorkerError } from "effect/unstable/workers";
import type { LayaEncoder, LayaRuntime, QtypeWord } from "laya-sdk";
import LayaWorker from "./laya.worker.ts?worker";
import type { InMsg, OutMsg } from "./layaWorkerProtocol";

type ProgressFn = (label: string, loaded?: number, total?: number) => void;

interface Pending {
  resolve: (msg: OutMsg) => void;
  reject: (err: Error) => void;
}

export interface LayaWorkerHandle {
  load: () => Promise<void>;
  runtime: LayaRuntime;
  dispose: () => void;
}

export function createLayaWorkerRuntime(onProgress: ProgressFn): LayaWorkerHandle {
  let seq = 0;
  const pending = new Map<number, Pending>();
  let fiber: Fiber.RuntimeFiber<never, unknown> | undefined;
  let workerRef: Worker.Worker<OutMsg, InMsg> | undefined;
  let dead = false;
  let booted = false;
  let notifySpawned!: () => void;
  const spawnedP = new Promise<void>((resolve) => {
    notifySpawned = resolve;
  });

  const failAll = (message: string): void => {
    const err = new Error(message);
    for (const [, p] of pending) p.reject(err);
    pending.clear();
  };

  const handleOut = (msg: OutMsg): Effect.Effect<void> =>
    Effect.sync(() => {
      if (msg.kind === "progress") {
        onProgress(msg.label, msg.loaded, msg.total);
        return;
      }
      const p = pending.get(msg.id);
      if (p === undefined) return;
      pending.delete(msg.id);
      if (msg.kind === "error") p.reject(new Error(msg.message));
      else p.resolve(msg);
    });

  // makePlatform returns a bare service value, not a Layer — lift it with
  // Layer.succeed (passing it straight to Effect.provide hangs the fiber).
  const PlatformLive = Layer.succeed(
    Worker.WorkerPlatform,
    Worker.makePlatform<Worker>()({
      // ponytail: NOT Effect.acquireRelease — in effect 4.0.0-rc.116 that
      // never completes the acquire here, so run() stalls before listen()
      // and nothing is ever sent to the worker. Scope.addFinalizer is the
      // working equivalent: dispose() interrupts run → scope closes → the
      // DedicatedWorker is terminated.
      setup: ({ worker, scope }) =>
        Scope.addFinalizer(scope, Effect.sync(() => worker.terminate())).pipe(Effect.as(worker)),
      listen: ({ port, emit, deferred }) =>
        Effect.sync(() => {
          port.onmessage = (event: MessageEvent) => emit(event.data);
          port.onerror = (event: Event) =>
            Effect.runFork(
              Deferred.fail(
                deferred,
                new WorkerError.WorkerError({
                  reason: new WorkerError.WorkerReceiveError({
                    message: `laya: worker error (${event instanceof ErrorEvent ? event.message : "unknown"})`,
                  }),
                })
              )
            );
        }),
    })
  );
  const SpawnerLive = Worker.layerSpawner((_id: number) => new LayaWorker());

  const boot = Effect.gen(function* () {
    const platform = yield* Worker.WorkerPlatform;
    const worker = yield* platform.spawn<OutMsg, InMsg>(0);
    workerRef = worker;
    yield* Effect.sync(notifySpawned);
    // No fork (this Effect version has no Effect.fork, and yielding a
    // non-Effect hangs the fiber silently): the boot fiber itself runs the
    // loop. Disposing interrupts boot, which unwinds run and terminates
    // the worker via finalizers. run() must pend for the worker's life;
    // a normal return means the channel is gone, so fail loudly — the
    // boot-level onExit below reports failures with their true cause.
    yield* worker.run(handleOut);
    dead = true;
    failAll("laya: worker stopped unexpectedly");
  }).pipe(Effect.provide(PlatformLive), Effect.provide(SpawnerLive));

  const bootOnce = (): void => {
    if (!booted) {
      booted = true;
      // Spawn happens inside boot, before the run-loop (and its ensuring
      // cleanup) exists — surface spawn failures or load() hangs forever.
      fiber = Effect.runFork(
        Effect.onExit(boot, (exit) =>
          Exit.isFailure(exit)
            ? Effect.sync(() => failAll(`laya: worker failed to start (${String(exit.cause).slice(0, 300)})`))
            : Effect.sync(() => {})
        )
      );
    }
  };

  const send = (msg: InMsg): Promise<void> => {
    if (dead) return Promise.reject(new Error("laya: worker stopped"));
    return spawnedP.then(() =>
      Effect.runPromise((workerRef as Worker.Worker<OutMsg, InMsg>).send(msg))
    );
  };

  const request = (msg: InMsg): Promise<OutMsg> =>
    new Promise<OutMsg>((resolve, reject) => {
      pending.set(msg.id, { resolve, reject });
      void send(msg).catch((e: unknown) => {
        pending.delete(msg.id);
        reject(e instanceof Error ? e : new Error(String(e)));
      });
    });

  const load = (): Promise<void> => {
    bootOnce();
    return request({ id: ++seq, kind: "load" }).then(() => undefined);
  };

  const score = (
    qtype: 0 | 1 | 2,
    qtypeWord: QtypeWord,
    instructions: string,
    options: string[],
    state: string
  ): Promise<number[]> =>
    request({ id: ++seq, kind: "score", args: { qtype, qtypeWord, instructions, options, state } }).then(
      (msg) => {
        if (msg.kind !== "result") throw new Error("laya: unexpected worker reply");
        return msg.probs;
      }
    );

  const runtime = {
    score,
    // SAFETY: encoder is never invoked on the worker path — the decisions
    // layer only calls score(), tokenizing happens inside the worker. The
    // cast keeps the shared LayaRuntime type without an SDK change.
    encoder: undefined as unknown as LayaEncoder,
  } as LayaRuntime;

  const dispose = (): void => {
    if (fiber !== undefined) {
      const f = fiber;
      fiber = undefined;
      void Effect.runPromise(Fiber.interrupt(f));
    }
  };

  return { load, runtime, dispose };
}
