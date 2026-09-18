/**
 * Thought-signature replay metadata. Gemini 3 attaches an opaque signature to
 * every tool call and requires it back on the follow-up request; the harness
 * keeps it beside the durable block in the adapter-private replay envelope so
 * message content stays provider-neutral.
 *
 * @module dsh-llm-vertex/replay
 */

import { LlmError } from '@deepseek-ai/dsh-llm'
import type { Message, ReplayEnvelope } from '@deepseek-ai/dsh-llm'

/** Envelope discriminator owned by this adapter. */
export const REPLAY_KIND = 'vertex-chat-completions'
/** Envelope layout version; bump on a structural change. */
export const REPLAY_VERSION = 1

/**
 * The marker Google documents for a tool call whose signature is unknown:
 * Vertex skips signature validation for that call instead of failing the
 * request, at the cost of the reasoning context that call carried.
 */
export const THOUGHT_SIGNATURE_SKIP_MARKER = 'skip_thought_signature_validator'

/** Index-aligned metadata retained alongside each emitted harness block. */
export interface ReplayBlock {
  type: 'text' | 'reasoning' | 'tool-call'
  /** Gemini thought signature; present only on tool-call blocks that carried one. */
  signature?: string
}

/**
 * Reject a non-object JSON value at provider and durable-data reads.
 * @param value - untrusted decoded JSON.
 * @param code - owning failure category.
 * @returns the validated object.
 */
export function object(value: unknown, code: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new LlmError('Vertex replay expected a JSON object', code)
  }
  return value as Record<string, unknown>
}

/**
 * Construct response metadata without duplicating the assistant text.
 * @param model - requested harness model id; signatures are not portable across models.
 * @param blocks - metadata in emitted block order.
 * @returns the versioned envelope persisted by the assembler.
 */
export function replayState(model: string, blocks: readonly ReplayBlock[]): ReplayEnvelope {
  return { response: { kind: REPLAY_KIND, version: REPLAY_VERSION, model }, blocks }
}

/**
 * Validate native replay, discarding unusable metadata before serializing durable content.
 * @param message - durable assistant content and source metadata.
 * @param model - target harness model id; cross-model signatures are not portable.
 * @param onDegrade - diagnostic for unusable metadata; receives no message content or signatures.
 * @returns index-aligned metadata, absent for foreign, cross-model, or degraded history.
 */
export function readReplay(message: Message, model: string, onDegrade?: (reason: string) => void): ReplayBlock[] | undefined {
  try {
    return validateReplay(message, model)
  } catch (error: unknown) {
    /* v8 ignore next -- the validator only throws INVALID_REPLAY_STATE; preserve future non-replay failures. */
    if (!(error instanceof LlmError) || error.code !== 'INVALID_REPLAY_STATE') throw error
    onDegrade?.(error.message)
    return undefined
  }
}

/* jscpd:ignore-start -- mirrors llm-deepseek replay validation; siblings stay symmetric until the protocol is shared */
function validateReplay(message: Message, model: string): ReplayBlock[] | undefined {
  if (message.source.kind !== 'model' || message.source.replayState === undefined) return undefined
  const fail = (detail: string): never => {
    throw new LlmError(`Vertex replay: ${detail}`, 'INVALID_REPLAY_STATE')
  }
  const envelope = object(message.source.replayState, 'INVALID_REPLAY_STATE')
  const response = object(envelope.response, 'INVALID_REPLAY_STATE')
  if (response.kind !== REPLAY_KIND || response.version !== REPLAY_VERSION) return fail('unsupported kind or version')
  if (response.model !== message.source.model) return fail('model does not match assistant source model')
  if (!Array.isArray(envelope.blocks) || envelope.blocks.length !== message.content.length) return fail('block count mismatch')
  const blocks = envelope.blocks.map((value, index): ReplayBlock => {
    const block = object(value, 'INVALID_REPLAY_STATE')
    if (block.type !== message.content[index]?.type
      || !['text', 'reasoning', 'tool-call'].includes(String(block.type))) return fail('block type mismatch')
    if (block.signature !== undefined && (block.type !== 'tool-call' || typeof block.signature !== 'string')) {
      return fail('invalid signature')
    }
    return block as unknown as ReplayBlock
  })
  return response.model === model ? blocks : undefined
}

/* jscpd:ignore-end */
