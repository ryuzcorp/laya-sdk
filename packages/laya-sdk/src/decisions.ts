import { Effect, Layer } from "effect";
import { DecisionModel } from "effect/unstable/ai";
import type { LayaRuntime } from "./runtime.js";
import { toClassifyAnswer, toProbabilityAnswer, toRateAnswer } from "./pure.js";

// Local DecisionModel: same Effect Decision shapes as a cloud provider,
// answered by a local Laya session instead.
export const makeLayaDecisionLive = (rt: LayaRuntime) =>
  Layer.effect(
    DecisionModel.DecisionModel,
    DecisionModel.make({
      decide: ({ state, decisions }) =>
        Effect.gen(function* () {
          const input = typeof state === "string" ? state : JSON.stringify(state);
          const answers: Record<string, DecisionModel.ProviderAnswer> = {};
          for (const [name, decision] of Object.entries(decisions)) {
            if (decision._tag === "Classify") {
              const labels = Object.keys(decision.criteria);
              const probs = yield* Effect.promise(() =>
                rt.score(0, "choice", decision.instructions, labels, input),
              );
              answers[name] = { _tag: "Classify", ...toClassifyAnswer(labels, probs) };
            } else if (decision._tag === "Probability") {
              const probs = yield* Effect.promise(() =>
                rt.score(2, "noul", decision.instructions, ["false", "true"], input),
              );
              answers[name] = { _tag: "Probability", ...toProbabilityAnswer(probs[1]) };
            } else {
              const levels = [...decision.criteria];
              const probs = yield* Effect.promise(() =>
                rt.score(1, "score", decision.instructions, levels, input),
              );
              answers[name] = { _tag: "Rate", ...toRateAnswer(levels, probs) };
            }
          }
          return {
            answers,
            usage: { inputTokens: undefined, outputTokens: undefined },
          } satisfies DecisionModel.ProviderResponse;
        }),
    }),
  );
