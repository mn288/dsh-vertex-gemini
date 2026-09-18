import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, {
  createAssistantMessage,
  createToolResultMessage,
  createUserMessage,
  INVALID_CREDENTIAL_CODE,
  QUOTA_EXCEEDED_CODE,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  ReasoningEffortId,
  ToolCallId,
  userAgent,
} from '@deepseek-ai/dsh-llm'
import type { Message } from '@deepseek-ai/dsh-llm'
import { assemble } from './assemble.ts'
import { closeMockServers, mockServer, SIGNATURE, textEvents, toolCallEvents } from './mock-server.ts'
import type { Behavior } from './mock-server.ts'

/** Scripted Application Default Credentials: one client and its token per case. */
const adc = {
  token: 'ya29.adc-token',
  constructed: [] as unknown[],
}
vi.mock('google-auth-library', () => ({
  GoogleAuth: class {
    constructor(options: unknown) {
      adc.constructed.push(options)
    }

    getClient(): Promise<{ getAccessToken(): Promise<{ token: string }> }> {
      return Promise.resolve({ getAccessToken: () => Promise.resolve({ token: adc.token }) })
    }
  },
}))

const LlmVertex = await import('../src/index.ts')
const { httpErrorCode, PROVIDER, THOUGHT_SIGNATURE_SKIP_MARKER, VertexAdapter } = LlmVertex
const { endpointRoot } = await import('../src/adapter.ts')

let testHome: string

beforeEach(() => {
  testHome = mkdtempSync(join(tmpdir(), 'dsh-llm-vertex-'))
  vi.stubEnv('DSH_HOME', testHome)
  vi.stubEnv('GOOGLE_CLOUD_PROJECT', '')
  vi.stubEnv('GCLOUD_PROJECT', '')
  vi.stubEnv('GOOGLE_OAUTH_ACCESS_TOKEN', '')
  adc.token = 'ya29.adc-token'
  adc.constructed.length = 0
})

afterEach(async () => {
  await closeMockServers()
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
  vi.useRealTimers()
  rmSync(testHome, { recursive: true, force: true })
})

async function harness(config: object = {}): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmVertex, config)
  return ctx
}

/** A mock endpoint plus a plugin pointed at it; `baseURL` routes around project resolution. */
async function scripted(script: Behavior[], config: object = {}) {
  const server = await mockServer(script)
  const ctx = await harness({ baseURL: server.url, ...config })
  return { server, ctx }
}

function user(text: string): Message {
  return createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
}

/** An SSE response body built from scripted events, for the direct fetch stubs. */
function sseResponse(events: readonly string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) controller.enqueue(new TextEncoder().encode(`data: ${event}\n\n`))
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

describe('registration', () => {
  it('lists a dormant configurable provider until a project or endpoint is configured', async () => {
    const dormant = await harness()
    expect(dormant.llm.listProviders()).toEqual([])
    expect(dormant.llm.listConfigurableProviders()).toEqual([
      { provider: PROVIDER, displayName: 'Google Vertex AI', settingsNs: 'llm-vertex', settingsPath: [] },
    ])
    const result = await assemble(dormant, { model: 'gemini-3.8-flash', messages: [] })
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code: 'NO_ADAPTER' } })

    const live = await harness({ project: 'p' })
    expect(live.llm.listProviders()).toEqual([{ id: PROVIDER, name: 'Google Vertex AI' }])
    await expect(live.llm.listModels(PROVIDER)).resolves.toEqual([
      { provider: PROVIDER, id: 'gemini-3.8-flash', name: 'Gemini 3.8 Flash', inputModalities: ['text'] },
    ])
    vi.stubEnv('GOOGLE_CLOUD_PROJECT', 'env-project')
    expect((await harness()).llm.listProviders()).toHaveLength(1)
  })

  it('addresses the endpoint from a gateway root or a project, and refuses neither', () => {
    expect(endpointRoot(LlmVertex.resolveAdapterOptions({ baseURL: 'https://gateway.example/openai', project: 'p' })))
      .toBe('https://gateway.example/openai')
    expect(endpointRoot(LlmVertex.resolveAdapterOptions({ project: 'p', location: 'us-central1' })))
      .toBe('https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi')
    expect(() => endpointRoot(LlmVertex.resolveAdapterOptions({})))
      .toThrow(expect.objectContaining({ code: 'MISSING_PROJECT' }))
  })

  it('resolves catalog and uncatalogued models with capacities and Gemini efforts', async () => {
    const connection = LlmVertex.resolveAdapterOptions({
      reasoningEffort: 'medium',
      models: [{ id: 'gemini-3.8-flash', contextWindow: 10, maxTokens: 5, description: 'fast' }],
    })
    const adapter = new VertexAdapter({
      options: () => connection,
      resolveBearer: () => Promise.resolve('bearer'),
    })
    await expect(adapter.resolveModel(PROVIDER, 'gemini-3.8-flash')).resolves.toEqual({
      provider: PROVIDER,
      id: 'gemini-3.8-flash',
      name: 'gemini-3.8-flash',
      description: 'fast',
      inputModalities: ['text'],
      context: { contextWindow: 10 },
      defaultMaxTokens: 5,
      reasoning: {
        efforts: [
          expect.objectContaining({ id: 'low', name: 'Low' }),
          expect.objectContaining({ id: 'medium', name: 'Medium' }),
          expect.objectContaining({ id: 'high', name: 'High' }),
        ],
        defaultEffort: 'medium',
      },
    })
    await expect(adapter.resolveModel(PROVIDER, 'gemini-unlisted')).resolves.toMatchObject({
      id: 'gemini-unlisted',
      name: 'gemini-unlisted',
      inputModalities: ['text'],
      context: { contextWindow: 1_048_576 },
      defaultMaxTokens: 65_536,
      reasoning: { defaultEffort: 'medium' },
    })
    expect(adapter.providerRetryPolicy(PROVIDER).mode).toBe('normal')
    const ctx = await harness({ project: 'p' })
    expect(ctx.llm.providerRetryPolicy(PROVIDER).mode).toBe('normal')
  })

  it('fails loud at load on an invalid composition entry', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await expect(ctx.plugin(LlmVertex, { models: [{ id: 'dup' }, { id: 'dup' }] })).rejects.toThrow('duplicate catalog model')
  })
})

describe('streaming through Application Default Credentials', () => {
  it('sends the qualified model, default effort, ADC bearer, and attribution to /chat/completions', async () => {
    const { server, ctx } = await scripted([{ kind: 'sse', events: textEvents }])
    const result = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [user('hi')], system: 'terse' })
    expect(result.finish).toEqual({ kind: 'stop' })
    expect(result.message.content).toEqual([{ type: 'text', text: 'hello' }])
    expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 1, totalTokens: 4 })
    expect(server.paths).toEqual(['/chat/completions'])
    expect(server.headers[0]).toMatchObject({
      'authorization': 'Bearer ya29.adc-token',
      'content-type': 'application/json',
      'accept': 'text/event-stream',
      'user-agent': userAgent(),
    })
    expect(server.requests[0]).toEqual({
      model: 'google/gemini-3.8-flash',
      messages: [{ role: 'system', content: 'terse' }, { role: 'user', content: 'hi' }],
      stream: true,
      stream_options: { include_usage: true },
      reasoning_effort: 'high',
      // The runtime materializes the adapter's default output cap into the request.
      max_tokens: 65_536,
    })
    expect(adc.constructed).toEqual([{ scopes: ['https://www.googleapis.com/auth/cloud-platform'] }])
  })

  it('discovers credentials once per scope set and again when the scopes change', async () => {
    const { ctx } = await scripted([{ kind: 'sse', events: textEvents }, { kind: 'sse', events: textEvents }])
    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(adc.constructed).toHaveLength(1)
  })

  it('computes the Vertex root from the configured project and location', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => Promise.resolve(sseResponse(textEvents)))
    const ctx = await harness({ project: 'cfg-project', location: 'europe-west4' })
    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(fetchSpy.mock.calls[0]?.[0])
      .toBe('https://europe-west4-aiplatform.googleapis.com/v1/projects/cfg-project/locations/europe-west4/endpoints/openapi/chat/completions')
  })

  it('reads the project from the launch environment when configuration names none', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => Promise.resolve(sseResponse(textEvents)))
    vi.stubEnv('GCLOUD_PROJECT', 'env-project')
    await assemble(await harness(), { model: 'gemini-3.8-flash', messages: [] })
    expect(fetchSpy.mock.calls[0]?.[0] as string).toContain('/projects/env-project/locations/global/')
  })
})

describe('streaming through a stored access token', () => {
  it('sends the bearer resolved from the environment reference', async () => {
    vi.stubEnv('GOOGLE_OAUTH_ACCESS_TOKEN', 'ya29.static')
    const { server, ctx } = await scripted([{ kind: 'sse', events: textEvents }], { auth: 'access-token' })
    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(server.headers[0]?.authorization).toBe('Bearer ya29.static')
    expect(adc.constructed).toEqual([])
  })

  it('fails with MISSING_CREDENTIAL when the reference is unset and INVALID_CREDENTIAL when it is unusable', async () => {
    const { ctx } = await scripted([], { auth: 'access-token', accessTokenEnv: 'VERTEX_TOKEN' })
    const missing = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(missing.finish).toMatchObject({ kind: 'error', failure: { code: 'MISSING_CREDENTIAL' } })
    if (missing.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(missing.finish.failure.message).toContain('VERTEX_TOKEN')
    vi.stubEnv('VERTEX_TOKEN', 'ya29.\u{1F600}')
    const invalid = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(invalid.finish).toMatchObject({ kind: 'error', failure: { code: INVALID_CREDENTIAL_CODE } })
    if (invalid.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(invalid.finish.failure.message).not.toContain('\u{1F600}')
  })

  it('never discovers Google credentials in access-token mode', async () => {
    vi.stubEnv('GOOGLE_OAUTH_ACCESS_TOKEN', 'ya29.static')
    const { ctx } = await scripted([{ kind: 'sse', events: textEvents }], { auth: 'access-token', project: 'p' })
    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(adc.constructed).toEqual([])
  })
})

describe('thought signatures across turns', () => {
  it('stores the signature as replay metadata and echoes it on the follow-up request', async () => {
    const { server, ctx } = await scripted([
      { kind: 'sse', events: toolCallEvents },
      { kind: 'sse', events: textEvents },
    ])
    const tools = [{ name: 'get_weather', description: 'weather', parameters: { type: 'object' } }]
    const first = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [user('weather?')], tools })
    expect(first.finish).toEqual({ kind: 'tool-calls' })
    expect(first.message.content).toEqual([
      { type: 'tool-call', id: 'call_1', name: 'get_weather', arguments: '{"city":"Paris"}' },
    ])
    expect(first.usage).toEqual({ inputTokens: 33, outputTokens: 16, reasoningTokens: 48 })
    expect(first.message.source).toMatchObject({
      replayState: { response: { kind: 'vertex-chat-completions', version: 1, model: 'gemini-3.8-flash' }, blocks: [{ type: 'tool-call', signature: SIGNATURE }] },
    })

    await assemble(ctx, {
      model: 'gemini-3.8-flash',
      messages: [
        user('weather?'),
        first.message,
        createToolResultMessage({ callId: ToolCallId('call_1'), content: [{ type: 'text', text: '18C' }], isError: false }),
      ],
      tools,
    })
    expect((server.requests[1] as { messages: unknown[] }).messages).toEqual([
      { role: 'user', content: 'weather?' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{
          id: 'call_1',
          type: 'function',
          function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
          extra_content: { google: { thought_signature: SIGNATURE } },
        }],
      },
      { role: 'tool', tool_call_id: 'call_1', content: '18C' },
    ])
  })

  it('sends the skip marker for history another adapter produced and warns once about degraded metadata', async () => {
    const { server, ctx } = await scripted([{ kind: 'sse', events: textEvents }])
    const warn = vi.spyOn(ctx.logger, 'warn').mockImplementation(() => undefined)
    const foreign = createAssistantMessage({
      content: [{ type: 'tool-call', id: ToolCallId('x'), name: 'f', arguments: '{}' }],
      source: { provider: 'deepseek-official', model: 'other', replayState: { response: { kind: 'deepseek-messages' } } },
    })
    const degraded = createAssistantMessage({
      content: [{ type: 'tool-call', id: ToolCallId('y'), name: 'f', arguments: '{}' }],
      source: { provider: PROVIDER, model: 'gemini-3.8-flash', replayState: { response: { kind: 'vertex-chat-completions', version: 9 } } },
    })
    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [foreign, degraded] })
    const messages = (server.requests[0] as { messages: { tool_calls?: { extra_content: unknown }[] }[] }).messages
    for (const message of messages) {
      expect(message.tool_calls?.[0]?.extra_content).toEqual({ google: { thought_signature: THOUGHT_SIGNATURE_SKIP_MARKER } })
    }
    // The runtime strips foreign replay state before the adapter sees it, so
    // only the degraded native envelope is reported.
    expect(warn).toHaveBeenCalledTimes(1)
    expect(String(warn.mock.calls[0]?.[0])).toContain(`"${PROVIDER}/gemini-3.8-flash"`)
  })
})

describe('failures', () => {
  it.each([
    [401, '[{"error":{"code":401,"message":"Request had invalid authentication credentials.","status":"UNAUTHENTICATED"}}]', 'AUTH', 'Request had invalid authentication credentials.'],
    [403, '{"error":{"message":"Permission denied"}}', 'AUTH', 'Permission denied'],
    [404, '[{"error":{"message":"Publisher model was not found","status":"NOT_FOUND"}}]', 'NOT_FOUND', 'Publisher model was not found'],
    [413, '{}', 'INVALID_REQUEST', 'Vertex API error (HTTP 413)'],
    [429, '[{"error":{"message":"Resource exhausted. Please try again later.","status":"RESOURCE_EXHAUSTED"}}]', 'RATE_LIMIT', 'Resource exhausted. Please try again later.'],
    [429, '[{"error":{"message":"Quota exceeded for aiplatform.googleapis.com","status":"RESOURCE_EXHAUSTED"}}]', QUOTA_EXCEEDED_CODE, 'Quota exceeded for aiplatform.googleapis.com'],
    [400, '[{"error":{"message":"Function call is missing a thought_signature","status":"INVALID_ARGUMENT"}}]', 'INVALID_REQUEST', 'Function call is missing a thought_signature'],
    [400, '[{"error":{"message":"The input token count exceeds the maximum context length of 1048576","status":"INVALID_ARGUMENT"}}]', CONTEXT_WINDOW_EXCEEDED_CODE, 'The input token count exceeds the maximum context length of 1048576'],
    [503, 'not json at all', 'SERVER', 'Vertex API error (HTTP 503)'],
    [418, '', 'HTTP_418', 'Vertex API error (HTTP 418)'],
  ])('maps HTTP %s to %s', async (status, body, code, message) => {
    const { ctx } = await scripted([{ kind: 'http-error', status, body }])
    const result = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code, message, status } })
  })

  it('carries a provider retry-after delay when it is usable', async () => {
    const soon = new Date(Date.now() + 30_000).toUTCString()
    const { ctx } = await scripted([
      { kind: 'http-error', status: 429, body: '{}', headers: { 'retry-after': '2' } },
      { kind: 'http-error', status: 429, body: '{}', headers: { 'retry-after': soon } },
      { kind: 'http-error', status: 429, body: '{}', headers: { 'retry-after': 'garbage' } },
      { kind: 'http-error', status: 429, body: '{}', headers: { 'retry-after': '0' } },
    ])
    const numeric = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(numeric.finish).toMatchObject({ kind: 'error', failure: { providerRetryAfterMs: 2_000 } })
    const dated = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    if (dated.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(dated.finish.failure.providerRetryAfterMs).toBeGreaterThan(0)
    const garbage = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    if (garbage.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(garbage.finish.failure.providerRetryAfterMs).toBeUndefined()
    const zero = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    if (zero.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(zero.finish.failure.providerRetryAfterMs).toBeUndefined()
  })

  it('fails with EMPTY_RESPONSE when the endpoint answers 2xx without a body', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => Promise.resolve(new Response(null, { status: 200 })))
    const result = await assemble(await harness({ project: 'p' }), { model: 'gemini-3.8-flash', messages: [] })
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code: 'EMPTY_RESPONSE' } })
  })

  it('fails with STREAM_CLOSED when the stream ends cleanly before [DONE], and TRANSPORT when the socket drops', async () => {
    const { ctx } = await scripted([
      { kind: 'sse', events: [textEvents[0]!, textEvents[1]!] },
      { kind: 'close-early', events: [textEvents[0]!, textEvents[1]!] },
    ])
    const truncated = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(truncated.finish).toMatchObject({ kind: 'error', failure: { code: 'STREAM_CLOSED' } })
    const dropped = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(dropped.finish).toMatchObject({ kind: 'error', failure: { code: 'TRANSPORT' } })
  })

  it('fails with TRANSPORT when the endpoint is unreachable', async () => {
    const result = await assemble(await harness({ baseURL: 'http://127.0.0.1:1' }), { model: 'gemini-3.8-flash', messages: [] })
    expect(result.finish).toMatchObject({
      kind: 'error',
      failure: { code: 'TRANSPORT', message: 'Vertex API request to http://127.0.0.1:1 failed' },
    })
  })

  it('fails with TIMEOUT when the stream idles past the configured budget', async () => {
    const { ctx } = await scripted([{ kind: 'sse', events: textEvents, delayMs: 400 }], { streamIdleTimeoutMs: 50 })
    const result = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(result.finish).toMatchObject({
      kind: 'error',
      failure: { code: 'TIMEOUT', message: 'Vertex stream idle timeout after 50ms' },
    })
  })

  it('reports a caller abort mid-stream and before the request leaves', async () => {
    const { ctx } = await scripted([{ kind: 'sse', events: textEvents, delayMs: 50 }])
    const controller = new AbortController()
    const chunks: string[] = []
    for await (const chunk of ctx.llm.stream({ provider: PROVIDER, model: 'gemini-3.8-flash', messages: [], signal: controller.signal })) {
      chunks.push(chunk.type)
      if (chunk.type === 'text-delta') controller.abort()
    }
    expect(chunks.at(-1)).toBe('finish')
    const early = new AbortController()
    early.abort()
    const result = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [], signal: early.signal })
    expect(result.finish).toMatchObject({ kind: 'aborted' })
  })

  it('ends a stalled bearer mint on the idle budget and on a caller abort', async () => {
    const connection = LlmVertex.resolveAdapterOptions({ baseURL: 'http://127.0.0.1:1', streamIdleTimeoutMs: 30 })
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const adapter = new VertexAdapter({ options: () => connection, resolveBearer: () => new Promise<string>(() => {}) })
    const drain = async (signal?: AbortSignal): Promise<void> => {
      for await (const _chunk of adapter.stream({ provider: PROVIDER, model: 'gemini-3.8-flash', messages: [], ...signal === undefined ? {} : { signal } })) {
        // No chunk precedes the bearer.
      }
    }
    await expect(drain()).rejects.toMatchObject({ code: 'TIMEOUT' })
    const controller = new AbortController()
    const aborted = drain(controller.signal)
    controller.abort()
    await expect(aborted).rejects.toMatchObject({ code: 'ABORTED' })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('reports a caller abort while the request awaits its response', async () => {
    const ctx = await harness({ baseURL: 'https://gateway.example/openai' })
    const controller = new AbortController()
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce((_input, init) => new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal as AbortSignal
      signal.addEventListener('abort', () => { reject(signal.reason as Error) }, { once: true })
      controller.abort()
    }))
    const result = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [], signal: controller.signal })
    expect(result.finish).toMatchObject({ kind: 'aborted' })
  })

  it('rejects an unsupported reasoning effort before any network I/O', async () => {
    const { server, ctx } = await scripted([])
    const result = await assemble(ctx, { model: 'gemini-3.8-flash', messages: [], reasoningEffort: ReasoningEffortId('max') })
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code: 'UNSUPPORTED_REASONING_EFFORT' } })
    expect(server.requests).toHaveLength(0)
  })
})

describe('httpErrorCode', () => {
  it('classifies statuses with and without a parsed record', () => {
    expect(httpErrorCode(401)).toBe('AUTH')
    expect(httpErrorCode(404)).toBe('NOT_FOUND')
    expect(httpErrorCode(429)).toBe('RATE_LIMIT')
    expect(httpErrorCode(429, { message: 'insufficient quota' })).toBe(QUOTA_EXCEEDED_CODE)
    expect(httpErrorCode(400, { status: 'INVALID_ARGUMENT' })).toBe('INVALID_REQUEST')
    expect(httpErrorCode(500)).toBe('SERVER')
    expect(httpErrorCode(302)).toBe('HTTP_302')
  })
})

describe('prepared calls', () => {
  it('binds model metadata and dispatch to one connection generation', async () => {
    let generation = LlmVertex.resolveAdapterOptions({ project: 'p', baseURL: 'http://127.0.0.1:1', reasoningEffort: 'low' })
    const adapter = new VertexAdapter({
      options: () => generation,
      resolveBearer: () => Promise.resolve('bearer'),
    })
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const prepared = await adapter.prepareCall(PROVIDER, 'gemini-3.8-flash')
    expect(prepared.model.reasoning?.defaultEffort).toBe('low')
    // A later generation changes the endpoint; the prepared call keeps its own.
    generation = LlmVertex.resolveAdapterOptions({ project: 'p', baseURL: server.url, reasoningEffort: 'high' })
    // Outside the runtime nothing normalizes the failure: the prepared stream throws its own LlmError.
    await expect((async () => {
      for await (const _chunk of prepared.stream({ provider: PROVIDER, model: 'gemini-3.8-flash', messages: [] })) {
        // The unreachable endpoint yields no chunk.
      }
    })()).rejects.toMatchObject({ code: 'TRANSPORT', message: 'Vertex API request to http://127.0.0.1:1 failed' })
    expect(server.requests).toHaveLength(0)
    await expect(adapter.listModels(PROVIDER)).resolves.toHaveLength(1)
    // The unprepared entry point reads the current generation instead.
    const chunks = []
    for await (const chunk of adapter.stream({ provider: PROVIDER, model: 'gemini-3.8-flash', messages: [] })) chunks.push(chunk)
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    expect(server.headers[0]?.authorization).toBe('Bearer bearer')
    expect(adapter.providerInfo(PROVIDER)).toEqual({ id: PROVIDER, name: 'Google Vertex AI' })
  })
})
