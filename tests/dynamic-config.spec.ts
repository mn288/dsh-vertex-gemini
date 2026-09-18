import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import LlmRuntime, { INVALID_CREDENTIAL_CODE } from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { LocalCredentialProvider } from '@deepseek-ai/dsh-credentials-local'
import { FileSettingsProvider } from '@deepseek-ai/dsh-settings-file'
import * as LlmVertex from '../src/index.ts'
import { assemble } from './assemble.ts'
import { closeMockServers, mockServer, textEvents } from './mock-server.ts'

const NS = 'llm-vertex'
const PROVIDER = LlmVertex.PROVIDER
const TOKEN_REF = credentialRef('GOOGLE_OAUTH_ACCESS_TOKEN')

const cleanups: Array<() => Promise<void>> = []

afterEach(async () => {
  while (cleanups.length > 0) await cleanups.pop()!()
  await closeMockServers()
  vi.unstubAllEnvs()
})

async function home(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'dsh-llm-vertex-dynamic-'))
  cleanups.push(() => rm(dir, { recursive: true, force: true }))
  return dir
}

interface Harness {
  ctx: Context
  settingsFiber: { dispose(): Promise<void> }
}

/**
 * Real dynamic composition: llm + settings-file + credentials-local +
 * llm-vertex over one temp harness home, authenticating with a stored access
 * token so no Google credential discovery runs in the test.
 */
async function boot(dir: string, config: object): Promise<Harness> {
  vi.stubEnv('DSH_HOME', dir)
  vi.stubEnv('GOOGLE_OAUTH_ACCESS_TOKEN', '')
  const ctx = new Context()
  cleanups.push(async () => {
    await ctx.fiber.dispose()
  })
  await ctx.plugin(LlmRuntime)
  const settingsFiber = ctx.plugin(FileSettingsProvider, { path: join(dir, 'settings.yaml'), watch: false })
  await settingsFiber
  await ctx.plugin(LocalCredentialProvider, { path: join(dir, '.credentials.yaml'), watch: false })
  await ctx.plugin(LlmVertex, { auth: 'access-token', ...config })
  return { ctx, settingsFiber }
}

function prompt(ctx: Context) {
  return assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
}

describe('request-level dynamic configuration', () => {
  it('routes the next request with the freshly resolved base URL and credential', async () => {
    const dir = await home()
    await writeFile(join(dir, '.credentials.yaml'), 'version: 1\nrefs:\n  GOOGLE_OAUTH_ACCESS_TOKEN: first-token\n', { mode: 0o600 })
    const serverA = await mockServer([{ kind: 'sse', events: textEvents }])
    const serverB = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx } = await boot(dir, { baseURL: serverA.url })

    await prompt(ctx)
    expect(serverA.headers[0]?.authorization).toBe('Bearer first-token')

    await ctx.settings.update(NS, { baseURL: serverB.url })
    await ctx.credentials.set(TOKEN_REF, 'second-token')

    await prompt(ctx)
    expect(serverA.requests).toHaveLength(1)
    expect(serverB.headers[0]?.authorization).toBe('Bearer second-token')
  })

  it('starts without a token and serves the next request once it is stored', async () => {
    const dir = await home()
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx } = await boot(dir, { baseURL: server.url })

    const missing = await prompt(ctx)
    expect(missing.finish).toMatchObject({ kind: 'error', failure: { code: 'MISSING_CREDENTIAL' } })
    await ctx.credentials.set(TOKEN_REF, 'ya29.arrived')
    await prompt(ctx)
    expect(server.headers[0]?.authorization).toBe('Bearer ya29.arrived')
  })

  it('rejects a stored token no header can carry, never echoing it in the failure', async () => {
    const dir = await home()
    const { ctx } = await boot(dir, { baseURL: 'http://127.0.0.1:1' })
    const secret = 'ya29.\u{1F600}supersecret'
    await ctx.credentials.set(TOKEN_REF, secret)
    const result = await prompt(ctx)
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code: INVALID_CREDENTIAL_CODE } })
    if (result.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(result.finish.failure.message).not.toContain('supersecret')
  })

  it('advertises a live settings catalog and project without re-registration', async () => {
    const dir = await home()
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx } = await boot(dir, { baseURL: server.url })

    await expect(ctx.llm.listModels(PROVIDER)).resolves.toHaveLength(1)
    await ctx.settings.update(NS, { models: [{ id: 'gemini-3.8-pro', name: 'From Settings' }], publisher: 'partner' })
    await expect(ctx.llm.listModels(PROVIDER)).resolves.toEqual([
      { provider: PROVIDER, id: 'gemini-3.8-pro', name: 'From Settings', inputModalities: ['text'] },
    ])
    await ctx.credentials.set(TOKEN_REF, 'ya29.t')
    await assemble(ctx, { model: 'gemini-3.8-pro', messages: [] })
    expect((server.requests[0] as { model: string }).model).toBe('partner/gemini-3.8-pro')
  })

  it('re-registers the route in place when the captured retry policy changes, without an empty-registry window', async () => {
    const dir = await home()
    const { ctx } = await boot(dir, { baseURL: 'http://127.0.0.1:1' })
    const observed: string[][] = []
    ctx.on('llm/adapters-updated', () => {
      observed.push(ctx.llm.listProviders().map(provider => provider.id))
    })

    await ctx.settings.update(NS, {
      retryPolicy: { mode: 'always', backoff: { initialDelayMs: 25, maxDelayMs: 100, jitterRatio: 0.2 } },
    })
    expect(ctx.llm.providerRetryPolicy(PROVIDER)).toEqual({
      mode: 'always',
      initialDelayMs: 25,
      maxDelayMs: 100,
      jitterRatio: 0.2,
    })
    expect(ctx.llm.listProviders()).toEqual([{ id: PROVIDER, name: 'Google Vertex AI' }])
    expect(observed).toEqual([[PROVIDER]])
    // An unrelated change leaves the registration alone.
    await ctx.settings.update(NS, { location: 'europe-west1' })
    expect(observed).toEqual([[PROVIDER]])
  })

  it('keeps the last good options when a settings snapshot fails beyond-schema validation', async () => {
    const dir = await home()
    const { ctx } = await boot(dir, { baseURL: 'http://127.0.0.1:1' })
    const errors = vi.spyOn(ctx.logger, 'error').mockImplementation(() => undefined)

    await ctx.settings.update(NS, { models: [{ id: 'dup' }, { id: 'dup' }] })
    await expect(ctx.llm.listModels(PROVIDER)).resolves.toHaveLength(1)
    expect(errors).toHaveBeenCalled()
    await ctx.settings.update(NS, { models: [{ id: 'recovered' }] })
    await expect(ctx.llm.listModels(PROVIDER)).resolves.toEqual([
      { provider: PROVIDER, id: 'recovered', name: 'recovered', inputModalities: ['text'] },
    ])
  })

  it('keeps the whole last-good snapshot when a rejected one changed the URL', async () => {
    const dir = await home()
    const good = await mockServer([{ kind: 'sse', events: textEvents }])
    const rejected = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx } = await boot(dir, { baseURL: good.url })
    vi.spyOn(ctx.logger, 'error').mockImplementation(() => undefined)
    await ctx.credentials.set(TOKEN_REF, 'ya29.good')

    await ctx.settings.update(NS, { baseURL: rejected.url, models: [{ id: 'dup' }, { id: 'dup' }] })
    await prompt(ctx)
    expect(rejected.requests).toHaveLength(0)
    expect(good.requests).toHaveLength(1)
    expect(good.headers[0]?.authorization).toBe('Bearer ya29.good')
  })

  it('falls back to the composition entry when settings detach', async () => {
    const dir = await home()
    await writeFile(join(dir, '.credentials.yaml'), 'version: 1\nrefs:\n  GOOGLE_OAUTH_ACCESS_TOKEN: steady\n', { mode: 0o600 })
    const serverA = await mockServer([{ kind: 'sse', events: textEvents }])
    const serverB = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx, settingsFiber } = await boot(dir, { baseURL: serverA.url })

    await ctx.settings.update(NS, { baseURL: serverB.url })
    await prompt(ctx)
    expect(serverB.requests).toHaveLength(1)

    await settingsFiber.dispose()
    await prompt(ctx)
    expect(serverA.requests).toHaveLength(1)
    expect(serverA.headers[0]?.authorization).toBe('Bearer steady')
  })

  it('registers the route when settings name a project and withdraws it when they stop, without an empty-registry window', async () => {
    const dir = await home()
    vi.stubEnv('GOOGLE_CLOUD_PROJECT', '')
    vi.stubEnv('GCLOUD_PROJECT', '')
    const { ctx } = await boot(dir, {})
    expect(ctx.llm.listProviders()).toEqual([])
    expect(ctx.llm.listConfigurableProviders().map(entry => entry.provider)).toEqual([PROVIDER])
    const observed: string[][] = []
    ctx.on('llm/adapters-updated', () => {
      observed.push(ctx.llm.listProviders().map(provider => provider.id))
    })

    await ctx.settings.update(NS, { project: 'from-settings' })
    expect(ctx.llm.listProviders()).toEqual([{ id: PROVIDER, name: 'Google Vertex AI' }])
    // An unrelated change leaves the registration alone.
    await ctx.settings.update(NS, { project: 'from-settings', location: 'europe-west1' })
    await ctx.settings.replace(NS, {})
    expect(ctx.llm.listProviders()).toEqual([])
    await ctx.settings.update(NS, { baseURL: 'http://127.0.0.1:1' })
    expect(ctx.llm.listProviders()).toHaveLength(1)
    expect(observed).toEqual([[PROVIDER], [], [PROVIDER]])
  })

  it('releases the route and its directory entry when the plugin is disposed', async () => {
    const dir = await home()
    const ctx = new Context()
    cleanups.push(async () => { await ctx.fiber.dispose() })
    vi.stubEnv('DSH_HOME', dir)
    await ctx.plugin(LlmRuntime)
    const fiber = ctx.plugin(LlmVertex, { auth: 'access-token', baseURL: 'http://127.0.0.1:1' })
    await fiber
    expect(ctx.llm.listProviders().map(provider => provider.id)).toEqual([PROVIDER])
    await fiber.dispose()
    expect(ctx.llm.listProviders()).toEqual([])
    expect(ctx.llm.listConfigurableProviders()).toEqual([])
  })
})
