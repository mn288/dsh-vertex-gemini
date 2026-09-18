/**
 * Translate Vertex OpenAI-compatible SSE payloads into the harness
 * `StreamChunk` protocol with one stateful block per content, reasoning, or
 * tool-call index. Gemini thought signatures are captured off each tool call's
 * first delta and emitted as index-aligned replay metadata on `finish`. Finish
 * reason and the latest usage are deferred until `[DONE]` so nothing follows
 * `finish`.
 *
 * @module dsh-llm-vertex/translate
 */

import { brandString } from '@deepseek-ai/dsh-brand'
import { EMPTY_RESPONSE_CODE, LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, FinishReason, StreamChunk, TokenUsage, ToolCallId } from '@deepseek-ai/dsh-llm'
import { DONE } from './sse.ts'
import { replayState } from './replay.ts'
import type { ReplayBlock } from './replay.ts'
import type { WireChunk, WireUsage } from './types.ts'

/* jscpd:ignore-start -- mirrors llm-deepseek chunk translation; siblings stay symmetric until the protocol is shared */
/** One open block under assembly. */
interface OpenBlock {
  index: number
  kind: 'text' | 'reasoning' | 'tool-call'
  text: string
  /** tool-call only, absent until a delta carries a non-empty value. */
  callId?: string | undefined
  name?: string | undefined
  /** tool-call only: the Gemini thought signature carried by this call. */
  signature?: string | undefined
}

/**
 * Map the wire finish_reason vocabulary to the harness FinishReason.
 * @param reason - the wire `finish_reason` string.
 * @returns the mapped reason; unrecognized values (content_filter, …) become `{kind: 'error'}` with the uppercased value as `code`.
 */
export function mapFinishReason(reason: string): FinishReason {
  switch (reason) {
    case 'stop': return { kind: 'stop' }
    case 'tool_calls': return { kind: 'tool-calls' }
    case 'length': return { kind: 'max-tokens' }
    default:
      return {
        kind: 'error',
        failure: { message: `model stopped: ${reason}`, code: reason.toUpperCase() },
      }
  }
}

/**
 * Map wire usage fields. Vertex's `prompt_tokens` INCLUDES cached tokens; the
 * harness TokenUsage convention is DISJOINT counts, so cache reads are
 * subtracted out of `inputTokens`.
 * @param usage - wire usage from the finish chunk or the trailing usage-only chunk.
 * @returns disjoint harness counts; an exact total is present only when the
 *   aggregate prompt/completion counters are valid and agree with any wire total.
 */
export function mapUsage(usage: WireUsage): TokenUsage {
  const cacheRead = usage.prompt_tokens_details?.cached_tokens
  const reasoning = usage.completion_tokens_details?.reasoning_tokens
  const combined = usage.prompt_tokens + usage.completion_tokens
  const hasExactTotal = Number.isSafeInteger(usage.prompt_tokens)
    && usage.prompt_tokens >= 0
    && Number.isSafeInteger(usage.completion_tokens)
    && usage.completion_tokens >= 0
    && (usage.total_tokens === undefined || usage.total_tokens === combined)
  return {
    inputTokens: usage.prompt_tokens - (cacheRead ?? 0),
    outputTokens: usage.completion_tokens,
    ...hasExactTotal ? { totalTokens: combined } : {},
    ...cacheRead !== undefined ? { cacheReadTokens: cacheRead } : {},
    ...reasoning !== undefined ? { reasoningTokens: reasoning } : {},
  }
}

/**
 * Accept one streamed identity field. The wire sends `id`, `name`, and the
 * signature once, on a call's first delta; a continuation that re-sends the
 * field empty or `null` means "no update", never "clear".
 */
function acceptIdentity(current: string | undefined, incoming: unknown): string | undefined {
  return typeof incoming === 'string' && incoming.length > 0 ? incoming : current
}

/** Assemble the final ContentBlock for one open block. */
function closeBlock(block: OpenBlock): ContentBlock {
  switch (block.kind) {
    case 'text': return { type: 'text', text: block.text }
    case 'reasoning': return { type: 'reasoning', text: block.text }
    case 'tool-call': return {
      type: 'tool-call',
      id: brandString<ToolCallId>(block.callId ?? ''),
      name: block.name ?? '',
      arguments: block.text,
    }
  }
}

/** Project one open block to its replay metadata entry. */
function replayBlock(block: OpenBlock): ReplayBlock {
  return {
    type: block.kind,
    ...block.signature === undefined ? {} : { signature: block.signature },
  }
}

/**
 * Consume SSE data payloads (ending with `[DONE]`) and yield StreamChunks.
 * Malformed JSON payloads abort the stream with `MALFORMED_RESPONSE`.
 * @param payloads - SSE data payloads from {@link parseSse}, `[DONE]`-terminated.
 * @param model - harness model id the request named; recorded in the replay envelope.
 * @returns deltas as they arrive; `block-end`s, `usage`, and `finish` are all deferred to the `[DONE]` sentinel.
 *   A `stop` (or absent) finish with no opened blocks maps to an `EMPTY_RESPONSE` error finish; a successful finish
 *   carries index-aligned thought-signature replay metadata.
 */
export async function* translate(payloads: AsyncIterable<string>, model: string): AsyncGenerator<StreamChunk> {
  let nextIndex = 0
  let textBlock: OpenBlock | undefined
  let reasoningBlock: OpenBlock | undefined
  const toolBlocks = new Map<number, OpenBlock>()
  const order: OpenBlock[] = []
  let pendingFinish: FinishReason | undefined
  let pendingUsage: TokenUsage | undefined

  function open(kind: OpenBlock['kind']): OpenBlock {
    const block: OpenBlock = { index: nextIndex++, kind, text: '' }
    order.push(block)
    return block
  }

  for await (const payload of payloads) {
    if (payload === DONE) {
      for (const block of order) {
        yield { type: 'block-end', index: block.index, block: closeBlock(block) }
      }
      if (pendingUsage) yield { type: 'usage', usage: pendingUsage }
      const reason = pendingFinish ?? { kind: 'stop' as const }
      if (reason.kind === 'stop' && order.length === 0) {
        yield {
          type: 'finish',
          reason: {
            kind: 'error',
            failure: { message: 'model returned a completed response with no content', code: EMPTY_RESPONSE_CODE },
          },
        }
        return
      }
      yield {
        type: 'finish',
        reason,
        ...reason.kind === 'error' ? {} : { replayState: replayState(model, order.map(replayBlock)) },
      }
      return
    }

    let chunk: WireChunk
    try {
      chunk = JSON.parse(payload) as WireChunk
    } catch {
      throw new LlmError(`malformed SSE payload: ${payload.slice(0, 120)}`, 'MALFORMED_RESPONSE')
    }

    for (const choice of chunk.choices ?? []) {
      const delta = choice.delta

      const reasoning = delta?.reasoning_content
      if (typeof reasoning === 'string' && reasoning.length > 0) {
        if (!reasoningBlock) {
          reasoningBlock = open('reasoning')
          yield { type: 'block-start', index: reasoningBlock.index, blockType: 'reasoning' }
        }
        reasoningBlock.text += reasoning
        yield { type: 'reasoning-delta', index: reasoningBlock.index, text: reasoning }
      }

      const content = delta?.content
      if (typeof content === 'string' && content.length > 0) {
        if (!textBlock) {
          textBlock = open('text')
          yield { type: 'block-start', index: textBlock.index, blockType: 'text' }
        }
        textBlock.text += content
        yield { type: 'text-delta', index: textBlock.index, text: content }
      }

      for (const call of delta?.tool_calls ?? []) {
        let block = toolBlocks.get(call.index)
        if (!block) {
          block = open('tool-call')
          toolBlocks.set(call.index, block)
          yield { type: 'block-start', index: block.index, blockType: 'tool-call' }
        }
        block.callId = acceptIdentity(block.callId, call.id)
        block.name = acceptIdentity(block.name, call.function?.name)
        block.signature = acceptIdentity(block.signature, call.extra_content?.google?.thought_signature)
        const fragment = call.function?.arguments ?? ''
        block.text += fragment
        yield {
          type: 'tool-call-delta',
          index: block.index,
          id: brandString<ToolCallId>(block.callId ?? ''),
          ...block.name !== undefined ? { name: block.name } : {},
          argumentsDelta: fragment,
        }
      }

      if (typeof choice.finish_reason === 'string') {
        pendingFinish = mapFinishReason(choice.finish_reason)
      }
    }

    if (chunk.usage) pendingUsage = mapUsage(chunk.usage)
  }

  // parseSse guarantees the [DONE] sentinel (or throws); reaching here means
  // the payload source violated that contract.
  throw new LlmError('SSE payload stream ended without [DONE]', 'STREAM_CLOSED')
}

/* jscpd:ignore-end */
