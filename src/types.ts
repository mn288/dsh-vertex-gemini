/**
 * Vertex AI OpenAI-compatible chat-completions wire format for Gemini. Types
 * only. Source of truth: Google Cloud "Call Gemini with the OpenAI library"
 * and "Thought signatures", cross-checked against live streams from the
 * `endpoints/openapi/chat/completions` path (2026-09).
 *
 * @module dsh-llm-vertex/types
 */

/** Request body for `POST {baseURL}/chat/completions`. */
export interface WireRequest {
  /** Publisher-qualified model id, for example `google/gemini-3.8-flash`. */
  model: string
  messages: WireMessage[]
  stream: true
  stream_options: { include_usage: true }
  /** Gemini thinking level; Vertex accepts exactly these three spellings. */
  reasoning_effort?: WireReasoningEffort
  tools?: WireTool[]
  temperature?: number
  max_tokens?: number
  /** Stop sequences (OpenAI `stop`), mapped from `GenerateOptions.stop`. */
  stop?: string[]
}

/** The effort vocabulary Vertex accepts on `reasoning_effort`. */
export type WireReasoningEffort = 'low' | 'medium' | 'high'

/** System-role message: a single string of instructions. */
export interface WireSystemMessage {
  role: 'system'
  content: string
}

/** User-role message: text only in this adapter. */
export interface WireUserMessage {
  role: 'user'
  content: string
}

/** Tool-role message: the result of one tool call, keyed by its call id. */
export interface WireToolMessage {
  role: 'tool'
  tool_call_id: string
  content: string
}

/**
 * Gemini-specific fields Vertex carries beside the OpenAI vocabulary. A
 * response tool call arrives with the signature filled in; a request tool call
 * must carry it back, or the documented skip marker when none is known.
 */
export interface WireGoogleExtraContent {
  google: { thought_signature: string }
}

/** A completed tool call replayed on an assistant history message; `arguments` is the raw JSON string. */
export interface WireToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
  /** Absent for a call Gemini left unsigned, such as the later calls of one parallel step. */
  extra_content?: WireGoogleExtraContent
}

/** Assistant-role history message. Text-less turns send `""`, never null. */
export interface WireAssistantMessage {
  role: 'assistant'
  content: string
  tool_calls?: WireToolCall[]
}

/** One entry of the request `messages` array, discriminated on `role`. */
export type WireMessage =
  | WireSystemMessage
  | WireUserMessage
  | WireAssistantMessage
  | WireToolMessage

/** One entry of the request `tools` array; `parameters` is a JSON Schema object. */
export interface WireTool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

/* jscpd:ignore-start -- mirrors llm-deepseek streamed chunk vocabulary; siblings stay symmetric until the protocol is shared */
/** One parsed SSE `data:` payload (a chat.completion.chunk). */
export interface WireChunk {
  choices?: WireChoice[]
  /** Arrives attached to the finish chunk or as a trailing usage-only chunk. */
  usage?: WireUsage | null
}

/** One streamed choice; `finish_reason` is non-null only on its terminal chunk. */
export interface WireChoice {
  delta?: WireDelta
  finish_reason?: string | null
}

/** The incremental content of one streamed choice. */
export interface WireDelta {
  role?: string
  /** Visible text; Vertex sends `""` on role-only and tool-call chunks. */
  content?: string | null
  /** Present when a deployment streams thoughts as reasoning text. */
  reasoning_content?: string | null
  tool_calls?: WireToolCallDelta[]
}

/** A streamed fragment of one tool call; fragments sharing an `index` concatenate into one call. */
export interface WireToolCallDelta {
  /** Disambiguates parallel tool calls; stable across a call's deltas. */
  index: number
  /** Carried by the first delta of each call. */
  id?: string | null
  type?: 'function'
  function?: {
    name?: string | null
    /** Argument JSON fragment (concatenate across deltas). */
    arguments?: string | null
  }
  /** Gemini thought signature, carried by the first delta of each call. */
  /* jscpd:ignore-end */
  extra_content?: Partial<WireGoogleExtraContent> | null
}
/* jscpd:ignore-end */

/**
 * Wire token accounting. Vertex reports `prompt_tokens` INCLUDING cached
 * tokens; `mapUsage` subtracts `prompt_tokens_details.cached_tokens` to keep
 * the harness convention of disjoint counts.
 */
export interface WireUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens?: number
  prompt_tokens_details?: { cached_tokens?: number }
  completion_tokens_details?: { reasoning_tokens?: number }
}

/** One Google API error record. */
export interface WireErrorRecord {
  error?: { code?: number; message?: string; status?: string }
}

/** Non-2xx error body: Vertex wraps the record in a one-element array; gateways may send it bare. */
export type WireError = WireErrorRecord | WireErrorRecord[]
