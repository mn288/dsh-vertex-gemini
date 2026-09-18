import { describe, expect, it, vi } from 'vitest'
import type { ImageBlock } from '@deepseek-ai/dsh-llm'
import {
  createAssistantMessage,
  createSystemMessage,
  createToolResultMessage,
  createUserMessage,
  ReasoningEffortId,
  ToolCallId,
} from '@deepseek-ai/dsh-llm'
import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import { serializeMessages, serializeRequest } from '../src/serialize.ts'
import { replayState, resolveAdapterOptions, THOUGHT_SIGNATURE_SKIP_MARKER } from '../src/index.ts'
import type { VertexConnectionOptions } from '../src/index.ts'

const MODEL = 'gemini-3.8-flash'
const connection: VertexConnectionOptions = resolveAdapterOptions({ project: 'p' })

function request(overrides: Partial<GenerateOptions>): GenerateOptions {
  return { provider: 'vertex-ai', model: MODEL, messages: [], ...overrides }
}

describe('serializeRequest', () => {
  it('builds a streaming request with the publisher-qualified model and the configured default effort', () => {
    expect(serializeRequest(request({ system: 'be terse' }), { connection })).toEqual({
      model: 'google/gemini-3.8-flash',
      messages: [{ role: 'system', content: 'be terse' }],
      stream: true,
      stream_options: { include_usage: true },
      reasoning_effort: 'high',
    })
  })

  it('maps tools, sampling, output cap, stop sequences, and a caller effort', () => {
    const body = serializeRequest(request({
      reasoningEffort: ReasoningEffortId('medium'),
      tools: [{ name: 't', description: 'd', parameters: { type: 'object' } }],
      temperature: 0.2,
      maxTokens: 100,
      stop: ['ZZZ'],
    }), { connection })
    expect(body).toMatchObject({
      reasoning_effort: 'medium',
      tools: [{ type: 'function', function: { name: 't', description: 'd', parameters: { type: 'object' } } }],
      temperature: 0.2,
      max_tokens: 100,
      stop: ['ZZZ'],
    })
    expect(serializeRequest(request({ tools: [] }), { connection })).not.toHaveProperty('tools')
  })

  it('forces the lowest effort for session titles and refuses efforts Vertex cannot spell', () => {
    expect(serializeRequest(request({ purpose: 'session-title', reasoningEffort: ReasoningEffortId('high') }), { connection }).reasoning_effort).toBe('low')
    expect(() => serializeRequest(request({ reasoningEffort: ReasoningEffortId('max') }), { connection }))
      .toThrow(expect.objectContaining({ code: 'UNSUPPORTED_REASONING_EFFORT' }))
  })

  it('passes a publisher-qualified model id through unchanged', () => {
    expect(serializeRequest(request({ model: 'meta/llama' }), { connection }).model).toBe('meta/llama')
  })
})

describe('serializeMessages', () => {
  it('flattens roles, expands tool results, and keeps empty tool output on the wire', () => {
    const wire = serializeMessages([
      createSystemMessage('sys', 'test'),
      createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } }),
      createUserMessage({
        content: [
          { type: 'text', text: 'mixed' },
          { type: 'tool-result', toolCallId: ToolCallId('c1'), content: [{ type: 'text', text: 'out' }] },
        ],
        source: { kind: 'user' },
      }),
      createToolResultMessage({ callId: ToolCallId('c2'), content: [], isError: false }),
    ], MODEL)
    expect(wire).toEqual([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
      { role: 'user', content: 'mixed' },
      { role: 'tool', tool_call_id: 'c1', content: 'out' },
      { role: 'tool', tool_call_id: 'c2', content: '(no output)' },
    ])
  })

  it('replays each tool call as Gemini signed it, omitting reasoning text', () => {
    const message = createAssistantMessage({
      content: [
        { type: 'reasoning', text: 'private' },
        { type: 'text', text: 'calling' },
        { type: 'tool-call', id: ToolCallId('c1'), name: 'f', arguments: '{"a":1}' },
        { type: 'tool-call', id: ToolCallId('c2'), name: 'g', arguments: '{}' },
      ],
      source: {
        provider: 'vertex-ai',
        model: MODEL,
        replayState: replayState(MODEL, [
          { type: 'reasoning' }, { type: 'text' }, { type: 'tool-call', signature: 'sig-1' }, { type: 'tool-call' },
        ]),
      },
    })
    expect(serializeMessages([message], MODEL)).toEqual([{
      role: 'assistant',
      content: 'calling',
      tool_calls: [
        { id: 'c1', type: 'function', function: { name: 'f', arguments: '{"a":1}' }, extra_content: { google: { thought_signature: 'sig-1' } } },
        { id: 'c2', type: 'function', function: { name: 'g', arguments: '{}' } },
      ],
    }])
  })

  it('sends the skip marker for foreign, cross-model, or degraded history and reports degradation', () => {
    const toolCall = { type: 'tool-call' as const, id: ToolCallId('c1'), name: 'f', arguments: '{}' }
    const foreign = createAssistantMessage({ content: [toolCall], source: { provider: 'other', model: 'x' } })
    const crossModel = createAssistantMessage({
      content: [toolCall],
      source: { provider: 'vertex-ai', model: 'gemini-old', replayState: replayState('gemini-old', [{ type: 'tool-call', signature: 'old' }]) },
    })
    const degraded = createAssistantMessage({
      content: [toolCall],
      source: { provider: 'vertex-ai', model: MODEL, replayState: { response: { kind: 'other' }, blocks: [] } },
    })
    const onDegrade = vi.fn()
    const wire = serializeMessages([foreign, crossModel, degraded], MODEL, onDegrade)
    for (const message of wire) {
      expect(message).toMatchObject({
        role: 'assistant',
        content: '',
        tool_calls: [expect.objectContaining({ extra_content: { google: { thought_signature: THOUGHT_SIGNATURE_SKIP_MARKER } } })],
      })
    }
    expect(onDegrade).toHaveBeenCalledTimes(1)
  })

  it('sends an assistant turn without tool calls as plain text only', () => {
    const message = createAssistantMessage({ content: [{ type: 'text', text: 'done' }], source: { provider: 'vertex-ai', model: MODEL } })
    expect(serializeMessages([message], MODEL)).toEqual([{ role: 'assistant', content: 'done' }])
  })

  it('refuses image content before it can be flattened away', () => {
    // The attachment identity is opaque to this adapter, which refuses the block before reading it.
    const attachment = { attachmentId: `sha256:${'a'.repeat(64)}`, mediaType: 'image/png', bytes: 3, width: 1, height: 1 } as unknown as ImageBlock['attachment']
    const image = createUserMessage({
      content: [{ type: 'image', attachment }],
      source: { kind: 'user' },
    })
    expect(() => serializeMessages([image], MODEL)).toThrow(expect.objectContaining({ code: 'UNSUPPORTED_CONTENT' }))
  })
})
