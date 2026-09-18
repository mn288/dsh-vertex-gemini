/** Model capabilities and reasoning choices for the `vertex-ai` route. */
import { ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type { LlmModelInfo, LlmResolvedModelInfo } from '@deepseek-ai/dsh-llm'
import type { VertexCatalogModel, VertexConnectionOptions, VertexReasoningEffort } from './config.ts'

const LOW_REASONING_EFFORT = ReasoningEffortId('low')
const MEDIUM_REASONING_EFFORT = ReasoningEffortId('medium')
const HIGH_REASONING_EFFORT = ReasoningEffortId('high')

/** Gemini thinking levels in escalation order, with selection guidance. */
export const REASONING_EFFORTS = [
  {
    id: LOW_REASONING_EFFORT,
    name: 'Low',
    description: 'Prefer for routine or latency-sensitive tasks.',
  },
  {
    id: MEDIUM_REASONING_EFFORT,
    name: 'Medium',
    description: 'A balance of depth and latency for everyday coding tasks.',
  },
  {
    id: HIGH_REASONING_EFFORT,
    name: 'High',
    description: 'The deepest level Gemini offers; the default for agentic work.',
  },
] as const

const DEFAULT_EFFORT_IDS = {
  low: LOW_REASONING_EFFORT,
  medium: MEDIUM_REASONING_EFFORT,
  high: HIGH_REASONING_EFFORT,
} as const satisfies Record<VertexReasoningEffort, ReasoningEffortId>

/**
 * Advertise one catalog entry.
 * @param provider - registered provider id.
 * @param model - advisory catalog entry.
 * @returns selector metadata.
 */
export function catalogModelInfo(provider: string, model: VertexCatalogModel): LlmModelInfo {
  return {
    provider,
    id: model.id,
    name: model.name ?? model.id,
    ...model.description === undefined ? {} : { description: model.description },
    inputModalities: ['text'],
  }
}

/**
 * Resolve model capabilities against one configuration generation. Every
 * route is text-only: declaring an unverified image capability would let the
 * host persist input the adapter rejects on every later turn.
 * @param connection - validated connection facts.
 * @param provider - registered provider id.
 * @param model - requested harness model id.
 * @returns effective model metadata for this operation.
 */
export function modelInfo(
  connection: VertexConnectionOptions,
  provider: string,
  model: string,
): LlmResolvedModelInfo {
  const configured = connection.models.find(entry => entry.id === model)
  return {
    ...configured === undefined
      ? { provider, id: model, name: model, inputModalities: ['text' as const] }
      : catalogModelInfo(provider, configured),
    context: { contextWindow: configured?.contextWindow ?? connection.defaultContextWindow },
    defaultMaxTokens: configured?.maxTokens ?? connection.maxTokens,
    reasoning: {
      efforts: REASONING_EFFORTS,
      defaultEffort: DEFAULT_EFFORT_IDS[connection.reasoningEffort],
    },
  }
}
