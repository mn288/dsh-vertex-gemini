/**
 * Serialize harness messages into a Vertex OpenAI-compatible chat-completions
 * request. Text-only: user content stays a string, tool results become
 * `role: tool` messages, and replayed tool calls carry the Gemini thought
 * signature they arrived with, or the documented skip marker when none is known.
 *
 * @module dsh-llm-vertex/serialize
 */

import { contentHasImage, LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message } from '@deepseek-ai/dsh-llm'
import { wireModelId } from './config.ts'
import type { VertexConnectionOptions, VertexReasoningEffort } from './config.ts'
import { readReplay, THOUGHT_SIGNATURE_SKIP_MARKER } from './replay.ts'
import type { ReplayBlock } from './replay.ts'
import type { WireMessage, WireReasoningEffort, WireRequest, WireTool, WireToolCall } from './types.ts'

/** Serializer inputs beyond the request itself. */
export interface SerializeContext {
  /** Connection facts of the same generation as the request. */
  connection: VertexConnectionOptions
  /** Diagnostic for unusable replay metadata; receives no content or signatures. */
  onReplayDegrade?: (reason: string) => void
}

/** Validate the adapter-owned effort before resolving its wire spelling. */
function reasoningEffort(effort: NonNullable<GenerateOptions['reasoningEffort']>): VertexReasoningEffort {
  if (effort === 'low' || effort === 'medium' || effort === 'high') return effort as VertexReasoningEffort
  throw new LlmError(
    `Vertex Gemini does not support reasoning effort "${effort}"`,
    'UNSUPPORTED_REASONING_EFFORT',
  )
}

/**
 * Resolve the wire effort. Session titles reserve output for visible text, so
 * they use the lowest level; other requests use the caller's effort or the
 * configured default.
 */
function resolveEffort(options: GenerateOptions, connection: VertexConnectionOptions): WireReasoningEffort {
  if (options.purpose === 'session-title') return 'low'
  return options.reasoningEffort === undefined
    ? connection.reasoningEffort
    : reasoningEffort(options.reasoningEffort)
}

/* jscpd:ignore-start -- mirrors llm-deepseek text projection; siblings stay symmetric until the protocol is shared */
/** Join the text blocks of a message (used for user/tool-result content). */
function flattenText(blocks: readonly ContentBlock[]): string {
  return blocks
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
}

/** Reject image content before any text-flattening path can silently erase it. */
function assertTextOnly(blocks: readonly ContentBlock[]): void {
  if (contentHasImage(blocks)) {
    throw new LlmError('The Vertex adapter does not support image content.', 'UNSUPPORTED_CONTENT')
  }
}

/* jscpd:ignore-end */
/**
 * Serialize one assistant message. Reasoning blocks have no passback field on
 * this wire and are omitted. With usable replay metadata a tool call carries
 * the signature Gemini sent for it, and a call Gemini left unsigned — every
 * call after the first of one parallel step — is sent unsigned again. Without
 * usable metadata (absent, foreign, or from another model) every call carries
 * the skip marker.
 */
function serializeAssistant(message: Message, replay: readonly ReplayBlock[] | undefined): WireMessage {
  const toolCalls: WireToolCall[] = []
  message.content.forEach((block, index) => {
    if (block.type !== 'tool-call') return
    const signature = replay === undefined ? THOUGHT_SIGNATURE_SKIP_MARKER : replay[index]?.signature
    toolCalls.push({
      id: block.id,
      type: 'function',
      function: { name: block.name, arguments: block.arguments },
      ...signature === undefined ? {} : { extra_content: { google: { thought_signature: signature } } },
    })
  })
  return {
    role: 'assistant',
    // Text-less turns send "" — never null; Vertex rejects null content.
    content: flattenText(message.content),
    ...toolCalls.length > 0 ? { tool_calls: toolCalls } : {},
  }
}

/**
 * Serialize the conversation. `tool-result` blocks become standalone
 * `{role: 'tool'}` messages; a mixed user message contributes its text first
 * and its tool results as separate wire messages after.
 * @param messages - the harness conversation, in order.
 * @param model - harness model id of this request; replay metadata from another model is not used.
 * @param onReplayDegrade - diagnostic for unusable replay metadata.
 * @returns the wire messages; order preserved, each tool result expanded into its own entry.
 */
/* jscpd:ignore-start -- mirrors llm-deepseek message serialization; siblings stay symmetric until the protocol is shared */
export function serializeMessages(
  messages: readonly Message[],
  model: string,
  onReplayDegrade?: (reason: string) => void,
): WireMessage[] {
  const wire: WireMessage[] = []
  for (const message of messages) {
    assertTextOnly(message.content)
    if (message.role === 'system') {
      wire.push({ role: 'system', content: flattenText(message.content) })
      continue
    }
    if (message.role === 'assistant') {
      wire.push(serializeAssistant(message, readReplay(message, model, onReplayDegrade)))
      continue
    }
    const toolResults = message.content.filter(block => block.type === 'tool-result')
    const text = flattenText(message.content)
    if (text.length > 0 || toolResults.length === 0) {
      wire.push({ role: 'user', content: text })
    }
    for (const result of toolResults) {
      wire.push({
        role: 'tool',
        tool_call_id: result.toolCallId,
        // Empty tool output still needs SOME content on the wire.
        content: flattenText(result.content) || '(no output)',
      })
    }
  }
  return wire
}

/* jscpd:ignore-end */
/**
 * Build the full wire request. Always streaming with usage reporting on;
 * optional fields are omitted rather than sent as null, so provider defaults apply.
 * @param options - the harness request (model, history, system, tools, sampling).
 * @param context - connection facts and the replay diagnostic.
 * @returns the chat-completions request body.
 */
export function serializeRequest(options: GenerateOptions, context: SerializeContext): WireRequest {
  const messages: WireMessage[] = []
  if (options.system !== undefined) {
    messages.push({ role: 'system', content: options.system })
  }
  messages.push(...serializeMessages(options.messages, options.model, context.onReplayDegrade))
  const tools: WireTool[] | undefined = options.tools?.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }))
  /* jscpd:ignore-start -- mirrors llm-deepseek request tail; siblings stay symmetric until the protocol is shared */
  return {
    model: wireModelId(options.model, context.connection.publisher),
    messages,
    stream: true,
    stream_options: { include_usage: true },
    reasoning_effort: resolveEffort(options, context.connection),
    ...tools !== undefined && tools.length > 0 ? { tools } : {},
    ...options.temperature !== undefined ? { temperature: options.temperature } : {},
    ...options.maxTokens === undefined ? {} : { max_tokens: options.maxTokens },
    ...options.stop !== undefined ? { stop: options.stop } : {},
  }
}

/* jscpd:ignore-end */
