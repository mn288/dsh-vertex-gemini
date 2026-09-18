/**
 * Real-composition guard for the dynamic-configuration chain: LlmRuntime,
 * settings-file, credentials-local, and llm-vertex boot from a test-only
 * cordis.yml through the actual Loader + Include path, external edits of
 * settings.yaml and the credentials document hot-publish through their
 * providers, and the very next request carries the fresh base URL and token.
 * The same composition without settings or credentials entries keeps
 * entry-config behavior — the documented optional-inject fallback.
 */

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import Include from '@deepseek-ai/cordis-plugin-include'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import LocalCredentialProvider from '@deepseek-ai/dsh-credentials-local'
import FileSettingsProvider from '@deepseek-ai/dsh-settings-file'
import * as LlmVertex from '../src/index.ts'
import { assemble } from './assemble.ts'
import { closeMockServers, mockServer, textEvents } from './mock-server.ts'

const NS = 'llm-vertex'
const TOKEN_REF = credentialRef('GOOGLE_OAUTH_ACCESS_TOKEN')

let root: string | undefined
let context: Context | undefined

afterEach(async () => {
  await context?.fiber.dispose()
  context = undefined
  if (root !== undefined) await rm(root, { recursive: true, force: true })
  root = undefined
  await closeMockServers()
  vi.unstubAllEnvs()
})

async function loadComposition(
  options: { withDynamic: boolean; baseURL: string },
): Promise<{ ctx: Context; settingsPath: string; credentialsPath: string }> {
  root = await mkdtemp(join(tmpdir(), 'dsh-llm-vertex-composition-'))
  vi.stubEnv('DSH_HOME', root)
  const settingsPath = join(root, 'settings.yaml')
  const credentialsPath = join(root, '.credentials.yaml')
  if (options.withDynamic) {
    await writeFile(settingsPath, '# personal settings\n')
    await writeFile(credentialsPath, 'version: 1\nrefs:\n  GOOGLE_OAUTH_ACCESS_TOKEN: boot-token\n', { mode: 0o600 })
  }

  const configPath = join(root, 'cordis.yml')
  await writeFile(configPath, [
    '- id: llm',
    "  name: '@deepseek-ai/dsh-llm'",
    ...options.withDynamic
      ? [
        '- id: settings',
        "  name: '@deepseek-ai/dsh-settings-file'",
        '  config:',
        `    path: ${JSON.stringify(settingsPath)}`,
        '    debounceMs: 10',
        '- id: credentials',
        "  name: '@deepseek-ai/dsh-credentials-local'",
        '  config:',
        `    path: ${JSON.stringify(credentialsPath)}`,
        '    debounceMs: 10',
      ]
      : [],
    '- id: llm-vertex',
    "  name: '@deepseek-ai/dsh-llm-vertex'",
    '  config:',
    '    auth: access-token',
    `    baseURL: ${JSON.stringify(options.baseURL)}`,
    '',
  ].join('\n'))

  const ctx = new Context()
  context = ctx
  ctx.baseUrl = pathToFileURL(root).href + '/'
  await ctx.plugin(Loader)
  ctx.loader.builtins.include = Include
  const modules = new Map<string, unknown>([
    ['@deepseek-ai/dsh-llm', LlmRuntime],
    ['@deepseek-ai/dsh-settings-file', FileSettingsProvider],
    ['@deepseek-ai/dsh-credentials-local', LocalCredentialProvider],
    ['@deepseek-ai/dsh-llm-vertex', LlmVertex],
  ])
  // The custom importer bypasses Node resolution; mirror the package manifests
  // a deployed cordis.yml has beside its declared dependencies.
  await Promise.all([...modules.keys()].map(async (packageName) => {
    const packageDir = join(root!, 'node_modules', ...packageName.split('/'))
    await mkdir(packageDir, { recursive: true })
    await writeFile(join(packageDir, 'package.json'), `${JSON.stringify({
      name: packageName,
      version: '0.1.0-rc.8',
      type: 'module',
    })}\n`)
  }))
  ctx.loader.internal = {
    version: 'v2',
    async import(specifier: string) {
      if (!modules.has(specifier)) throw new Error(`unexpected Loader import: ${specifier}`)
      return modules.get(specifier)
    },
  } as unknown as NonNullable<typeof ctx.loader.internal>
  await ctx.loader.create({
    name: 'cordis:include',
    config: { path: pathToFileURL(configPath).href },
  })
  await ctx.loader.await()
  return { ctx, settingsPath, credentialsPath }
}

describe('llm-vertex real dynamic composition', () => {
  it('boots from cordis.yml and routes the next request after external settings and credential edits', async () => {
    vi.stubEnv('GOOGLE_OAUTH_ACCESS_TOKEN', '')
    const serverA = await mockServer([{ kind: 'sse', events: textEvents }])
    const serverB = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx, settingsPath, credentialsPath } = await loadComposition({ withDynamic: true, baseURL: serverA.url })

    expect(ctx.get('settings')!.describe().map(entry => entry.ns)).toEqual([NS])
    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(serverA.headers[0]?.authorization).toBe('Bearer boot-token')

    await writeFile(settingsPath, `llm-vertex:\n  baseURL: ${serverB.url}\n`)
    await vi.waitFor(() => {
      expect((ctx.get('settings')!.get(NS) as { baseURL?: string }).baseURL).toBe(serverB.url)
    }, { timeout: 5000 })
    await writeFile(credentialsPath, 'version: 1\nrefs:\n  GOOGLE_OAUTH_ACCESS_TOKEN: rotated-token\n', { mode: 0o600 })
    await vi.waitFor(async () => {
      expect(await ctx.get('credentials')!.resolve(TOKEN_REF)).toEqual({ value: 'rotated-token', source: 'file' })
    }, { timeout: 5000 })

    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(serverA.requests).toHaveLength(1)
    expect(serverB.headers[0]?.authorization).toBe('Bearer rotated-token')
  })

  it('boots the same adapter on entry config alone, resolving the reference from the environment', async () => {
    vi.stubEnv('GOOGLE_OAUTH_ACCESS_TOKEN', 'entry-token')
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx } = await loadComposition({ withDynamic: false, baseURL: server.url })

    expect(ctx.get('settings')).toBeUndefined()
    expect(ctx.get('credentials')).toBeUndefined()
    await assemble(ctx, { model: 'gemini-3.8-flash', messages: [] })
    expect(server.headers[0]?.authorization).toBe('Bearer entry-token')
  })
})
