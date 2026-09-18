import { describe, expect, it } from 'vitest'
import { EMPTY_RESPONSE_CODE, LlmError } from '@deepseek-ai/dsh-llm'
import type { StreamChunk } from '@deepseek-ai/dsh-llm'
import { mapFinishReason, mapUsage, translate } from '../src/translate.ts'
import { REPLAY_KIND } from '../src/index.ts'
import { SIGNATURE, textEvents, toolCallEvents } from './mock-server.ts'

async function* payloads(events: readonly string[]): AsyncGenerator<string> {
  for (const event of events) yield event
}

async function collect(events: readonly string[], model = 'gemini-3.8-flash'): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of translate(payloads(events), model)) chunks.push(chunk)
  return chunks
}

describe('mapFinishReason', () => {
  it('maps the wire vocabulary and turns unknown reasons into errors', () => {
    expect(mapFinishReason('stop')).toEqual({ kind: 'stop' })
    expect(mapFinishReason('tool_calls')).toEqual({ kind: 'tool-calls' })
    expect(mapFinishReason('length')).toEqual({ kind: 'max-tokens' })
    expect(mapFinishReason('content_filter')).toEqual({
      kind: 'error',
      failure: { message: 'model stopped: content_filter', code: 'CONTENT_FILTER' },
    })
  })
})

describe('mapUsage', () => {
  it('keeps disjoint counts and derives an exact total when the wire agrees', () => {
    expect(mapUsage({ prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 })).toEqual({
      inputTokens: 10, outputTokens: 4, totalTokens: 14,
    })
    expect(mapUsage({
      prompt_tokens: 10,
      completion_tokens: 4,
      prompt_tokens_details: { cached_tokens: 6 },
      completion_tokens_details: { reasoning_tokens: 3 },
    })).toEqual({ inputTokens: 4, outputTokens: 4, totalTokens: 14, cacheReadTokens: 6, reasoningTokens: 3 })
  })

  it('omits the total when the wire total disagrees or a counter is invalid', () => {
    expect(mapUsage({ prompt_tokens: 10, completion_tokens: 4, total_tokens: 97 })).toEqual({ inputTokens: 10, outputTokens: 4 })
    expect(mapUsage({ prompt_tokens: -1, completion_tokens: 4 })).toEqual({ inputTokens: -1, outputTokens: 4 })
    expect(mapUsage({ prompt_tokens: 1.5, completion_tokens: 4 })).toEqual({ inputTokens: 1.5, outputTokens: 4 })
    expect(mapUsage({ prompt_tokens: 1, completion_tokens: -4 })).toEqual({ inputTokens: 1, outputTokens: -4 })
  })
})

describe('translate', () => {
  it('streams text, defers block-end, usage, and finish to [DONE], and carries replay metadata', async () => {
    const chunks = await collect(textEvents)
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'hello' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'hello' } },
      { type: 'usage', usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 } },
      {
        type: 'finish',
        reason: { kind: 'stop' },
        replayState: { response: { kind: REPLAY_KIND, version: 1, model: 'gemini-3.8-flash' }, blocks: [{ type: 'text' }] },
      },
    ])
  })

  it('captures the thought signature of a tool call into aligned replay metadata', async () => {
    const chunks = await collect(toolCallEvents)
    expect(chunks[0]).toEqual({ type: 'block-start', index: 0, blockType: 'tool-call' })
    expect(chunks[1]).toEqual({
      type: 'tool-call-delta', index: 0, id: 'call_1', name: 'get_weather', argumentsDelta: '{"city":"Paris"}',
    })
    expect(chunks[2]).toEqual({
      type: 'block-end',
      index: 0,
      block: { type: 'tool-call', id: 'call_1', name: 'get_weather', arguments: '{"city":"Paris"}' },
    })
    expect(chunks[3]).toEqual({
      type: 'usage',
      usage: { inputTokens: 33, outputTokens: 16, reasoningTokens: 48 },
    })
    expect(chunks[4]).toEqual({
      type: 'finish',
      reason: { kind: 'tool-calls' },
      replayState: {
        response: { kind: REPLAY_KIND, version: 1, model: 'gemini-3.8-flash' },
        blocks: [{ type: 'tool-call', signature: SIGNATURE }],
      },
    })
  })

  it('assembles interleaved reasoning, text, and parallel tool calls across fragments', async () => {
    const chunks = await collect([
      '{"choices":[{"delta":{"reasoning_content":""}}]}',
      '{"choices":[{"delta":{"reasoning_content":"think"}}]}',
      '{"choices":[{"delta":{"content":"a"}}]}',
      '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"f","arguments":"{\\"a\\""}},{"index":1,"id":"c1","function":{"name":"g","arguments":""}}]}}]}',
      '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"","function":{"name":null,"arguments":":1}"},"extra_content":null}]}}]}',
      '{"choices":[{"delta":{"content":"b"},"finish_reason":"tool_calls"}]}',
      '[DONE]',
    ])
    const ends = chunks.filter(chunk => chunk.type === 'block-end')
    expect(ends.map(chunk => chunk.type === 'block-end' ? chunk.block : undefined)).toEqual([
      { type: 'reasoning', text: 'think' },
      { type: 'text', text: 'ab' },
      { type: 'tool-call', id: 'c0', name: 'f', arguments: '{"a":1}' },
      { type: 'tool-call', id: 'c1', name: 'g', arguments: '' },
    ])
    expect(chunks.filter(chunk => chunk.type === 'usage')).toEqual([])
    const finish = chunks.at(-1)
    expect(finish).toMatchObject({ type: 'finish', reason: { kind: 'tool-calls' } })
    expect(finish?.type === 'finish' ? finish.replayState?.blocks : undefined).toEqual([
      { type: 'reasoning' }, { type: 'text' }, { type: 'tool-call' }, { type: 'tool-call' },
    ])
    // The first delta of a tool call announces its identity to the loop.
    expect(chunks.filter(chunk => chunk.type === 'tool-call-delta').map(chunk => chunk.type === 'tool-call-delta' ? [chunk.id, chunk.name, chunk.argumentsDelta] : [])).toEqual([
      ['c0', 'f', '{"a"'], ['c1', 'g', ''], ['c0', 'f', ':1}'],
    ])
  })

  it('closes an identity-less tool call with empty id and name and defaults a missing finish reason to stop', async () => {
    const chunks = await collect([
      '{"choices":[{"delta":{"reasoning_content":"a"}}]}',
      '{"choices":[{"delta":{"reasoning_content":"b"}}]}',
      '{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}',
      '{"choices":[{"delta":{"tool_calls":[{"index":0,"extra_content":{},"function":{"arguments":"{}"}}]}}]}',
      '{"choices":[{}]}',
      '[DONE]',
    ])
    expect(chunks.filter(chunk => chunk.type === 'tool-call-delta')).toEqual([
      { type: 'tool-call-delta', index: 1, id: '', argumentsDelta: '' },
      { type: 'tool-call-delta', index: 1, id: '', argumentsDelta: '{}' },
    ])
    expect(chunks.filter(chunk => chunk.type === 'block-end').map(chunk => chunk.type === 'block-end' ? chunk.block : undefined)).toEqual([
      { type: 'reasoning', text: 'ab' },
      { type: 'tool-call', id: '', name: '', arguments: '{}' },
    ])
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
  })

  it('ignores chunks without choices and role-only deltas', async () => {
    const chunks = await collect([
      '{"id":"x"}',
      '{"choices":[{"delta":{"role":"assistant","content":null}}]}',
      '{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
      '[DONE]',
    ])
    expect(chunks.map(chunk => chunk.type)).toEqual(['block-start', 'text-delta', 'block-end', 'finish'])
  })

  it('turns a completed response with no content into an EMPTY_RESPONSE error finish without replay metadata', async () => {
    const chunks = await collect([
      '{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}',
      '[DONE]',
    ])
    expect(chunks).toEqual([{
      type: 'finish',
      reason: { kind: 'error', failure: { message: 'model returned a completed response with no content', code: EMPTY_RESPONSE_CODE } },
    }])
  })

  it('keeps a max-tokens finish and an unknown-reason error finish, the latter without replay metadata', async () => {
    const cut = await collect(['{"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}', '[DONE]'])
    expect(cut.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'max-tokens' } })
    expect(cut.at(-1)).toHaveProperty('replayState')
    const filtered = await collect(['{"choices":[{"delta":{"content":"partial"},"finish_reason":"content_filter"}]}', '[DONE]'])
    expect(filtered.at(-1)).toEqual({
      type: 'finish',
      reason: { kind: 'error', failure: { message: 'model stopped: content_filter', code: 'CONTENT_FILTER' } },
    })
  })

  it('fails loud on malformed payloads and on a payload source that ends without [DONE]', async () => {
    await expect(collect(['{not json'])).rejects.toMatchObject({ code: 'MALFORMED_RESPONSE' })
    await expect(collect(['{"choices":[]}'])).rejects.toBeInstanceOf(LlmError)
    await expect(collect(['{"choices":[]}'])).rejects.toMatchObject({ code: 'STREAM_CLOSED' })
  })
})
