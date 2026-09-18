import { describe, expect, it, vi } from 'vitest'
import { createAssistantMessage, createUserMessage, LlmError, ToolCallId } from '@deepseek-ai/dsh-llm'
import type { Message } from '@deepseek-ai/dsh-llm'
import { readReplay, replayState, REPLAY_KIND, REPLAY_VERSION } from '../src/index.ts'
import { object } from '../src/replay.ts'

const MODEL = 'gemini-3.8-flash'

function assistant(replay: unknown, model = MODEL): Message {
  return createAssistantMessage({
    content: [
      { type: 'text', text: 'calling' },
      { type: 'tool-call', id: ToolCallId('call_1'), name: 'get_weather', arguments: '{}' },
    ],
    source: { provider: 'vertex-ai', model, ...replay === undefined ? {} : { replayState: replay } },
  })
}

describe('replay envelope', () => {
  it('builds the versioned envelope beside the blocks', () => {
    expect(replayState(MODEL, [{ type: 'text' }, { type: 'tool-call', signature: 'sig' }])).toEqual({
      response: { kind: REPLAY_KIND, version: REPLAY_VERSION, model: MODEL },
      blocks: [{ type: 'text' }, { type: 'tool-call', signature: 'sig' }],
    })
  })

  it('reads back aligned metadata for the same model and nothing for another model', () => {
    const message = assistant(replayState(MODEL, [{ type: 'text' }, { type: 'tool-call', signature: 'sig' }]))
    expect(readReplay(message, MODEL)).toEqual([{ type: 'text' }, { type: 'tool-call', signature: 'sig' }])
    expect(readReplay(message, 'gemini-other')).toBeUndefined()
  })

  it('answers nothing for foreign or replay-less history', () => {
    expect(readReplay(assistant(undefined), MODEL)).toBeUndefined()
    expect(readReplay(createUserMessage({ content: [], source: { kind: 'user' } }), MODEL)).toBeUndefined()
  })

  it.each([
    ['a non-object envelope', 'nope'],
    ['a non-object response', { response: 1, blocks: [] }],
    ['another kind', { response: { kind: 'other', version: 1, model: MODEL }, blocks: [] }],
    ['another version', { response: { kind: REPLAY_KIND, version: 2, model: MODEL }, blocks: [] }],
    ['a mismatched source model', { response: { kind: REPLAY_KIND, version: 1, model: 'x' }, blocks: [] }],
    ['a block count mismatch', { response: { kind: REPLAY_KIND, version: 1, model: MODEL }, blocks: [{ type: 'text' }] }],
    ['a non-object block', { response: { kind: REPLAY_KIND, version: 1, model: MODEL }, blocks: [1, 2] }],
    ['a block type mismatch', { response: { kind: REPLAY_KIND, version: 1, model: MODEL }, blocks: [{ type: 'tool-call' }, { type: 'text' }] }],
    ['a signature on a text block', { response: { kind: REPLAY_KIND, version: 1, model: MODEL }, blocks: [{ type: 'text', signature: 's' }, { type: 'tool-call' }] }],
    ['a non-string signature', { response: { kind: REPLAY_KIND, version: 1, model: MODEL }, blocks: [{ type: 'text' }, { type: 'tool-call', signature: 7 }] }],
  ])('degrades %s and reports why without content', (_label, replay) => {
    const onDegrade = vi.fn()
    expect(readReplay(assistant(replay), MODEL, onDegrade)).toBeUndefined()
    expect(onDegrade).toHaveBeenCalledTimes(1)
    const reason = onDegrade.mock.calls[0]?.[0] as string
    expect(reason).toMatch(/^Vertex replay/)
    expect(reason).not.toContain('calling')
  })

  it('rejects non-objects at durable reads with the owning code', () => {
    expect(() => object([], 'INVALID_REPLAY_STATE')).toThrow(LlmError)
    expect(() => object(null, 'X')).toThrow(expect.objectContaining({ code: 'X' }))
    expect(object({ a: 1 }, 'X')).toEqual({ a: 1 })
  })
})
