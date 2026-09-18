/**
 * Decode a Vertex SSE byte stream into event `data` payloads. Framing is
 * `eventsource-parser`'s; this module keeps the OpenAI-compatible protocol:
 * the literal `[DONE]` is yielded so the caller owns final flushing, and EOF
 * before it raises {@link LlmError}.
 *
 * @module dsh-llm-vertex/sse
 */

/* jscpd:ignore-start -- mirrors llm-deepseek SSE framing; siblings stay symmetric until the protocol is shared */
import { EventSourceParserStream } from 'eventsource-parser/stream'
import { LlmError } from '@deepseek-ai/dsh-llm'

/** The terminal payload Vertex sends after the last chunk. */
export const DONE = '[DONE]'

/**
 * Parse an SSE byte stream into data payloads. Yields `[DONE]` as the final
 * value and returns; throws `LlmError('STREAM_CLOSED')` when the stream ends
 * without it (truncated response — the model call cannot be trusted).
 * @param stream - raw SSE bytes; reads may split anywhere, including mid-UTF-8 sequence.
 * @param onComment - optional transport-activity callback; comments never enter the yielded payload stream.
 * @returns each event's data payload in arrival order, the `[DONE]` sentinel last.
 */
export async function* parseSse(
  stream: ReadableStream<BufferSource>,
  onComment?: (comment: string) => void,
): AsyncGenerator<string> {
  const events = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream({ onComment }))
  for await (const { data } of events) {
    yield data
    if (data === DONE) return
  }
  throw new LlmError('Vertex SSE stream ended without [DONE]', 'STREAM_CLOSED')
}

/* jscpd:ignore-end */
