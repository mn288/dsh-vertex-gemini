/**
 * Real-API check against Vertex AI Gemini. Runs only when `DSH_VERTEX_E2E=1`
 * and Application Default Credentials are available (`gcloud auth
 * application-default login`); the project comes from `DSH_VERTEX_PROJECT`,
 * `GOOGLE_CLOUD_PROJECT`, or `GCLOUD_PROJECT`. The composition is the same
 * entry the shipped bundle mounts, plus the project a user would enter.
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createToolResultMessage, createUserMessage, ToolCallId } from '@deepseek-ai/dsh-llm'
import * as LlmVertex from '../src/index.ts'
import { assemble } from './assemble.ts'

const ENABLED = process.env['DSH_VERTEX_E2E'] === '1'
const MODEL = process.env['DSH_VERTEX_MODEL'] ?? 'gemini-3.8-flash'
const PROJECT = process.env['DSH_VERTEX_PROJECT']

let home: string
let ctx: Context

beforeAll(async () => {
  home = mkdtempSync(join(tmpdir(), 'dsh-llm-vertex-e2e-'))
  vi.stubEnv('DSH_HOME', home)
  ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmVertex, PROJECT === undefined ? {} : { project: PROJECT })
})

afterAll(async () => {
  await ctx?.fiber.dispose()
  vi.unstubAllEnvs()
  rmSync(home, { recursive: true, force: true })
})

describe.skipIf(!ENABLED)('llm-vertex against Vertex AI', () => {
  it('streams a text answer with usage through Application Default Credentials', async () => {
    const result = await assemble(ctx, {
      model: MODEL,
      messages: [createUserMessage({ content: [{ type: 'text', text: 'Reply with exactly: pong' }], source: { kind: 'user' } })],
      maxTokens: 64,
    })
    expect(result.finish).toEqual({ kind: 'stop' })
    const text = result.message.content[0]
    if (text?.type !== 'text') throw new Error('expected a text block')
    expect(text.text).toContain('pong')
    expect(result.usage?.outputTokens).toBeGreaterThan(0)
  }, 120_000)

  it('completes a tool call round trip with the thought signature replayed', async () => {
    const tools = [{
      name: 'get_weather',
      description: 'Get the weather for a city',
      parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
    }]
    const ask = createUserMessage({ content: [{ type: 'text', text: 'What is the weather in Paris? Use the tool.' }], source: { kind: 'user' } })
    const first = await assemble(ctx, { model: MODEL, messages: [ask], tools, maxTokens: 512 })
    expect(first.finish).toEqual({ kind: 'tool-calls' })
    const call = first.message.content.find(block => block.type === 'tool-call')
    if (call === undefined || call.type !== 'tool-call') throw new Error('expected a tool call')
    expect(call.name).toBe('get_weather')
    if (first.message.source.kind !== 'model') throw new Error('expected a model source')
    const replay = first.message.source.replayState as { blocks: { type: string; signature?: string }[] }
    const signed = replay.blocks.find(block => block.type === 'tool-call')
    expect(typeof signed?.signature).toBe('string')

    const second = await assemble(ctx, {
      model: MODEL,
      messages: [
        ask,
        first.message,
        createToolResultMessage({ callId: ToolCallId(call.id), content: [{ type: 'text', text: '{"temp_c":18,"sky":"cloudy"}' }], isError: false }),
      ],
      tools,
      maxTokens: 256,
    })
    // Without the signature Vertex answers HTTP 400; a stop finish is the round trip succeeding.
    expect(second.finish).toEqual({ kind: 'stop' })
    const answer = second.message.content[0]
    if (answer?.type !== 'text') throw new Error('expected a text block')
    expect(answer.text).toMatch(/18/)
  }, 180_000)
})
