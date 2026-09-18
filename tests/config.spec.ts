import { describe, expect, it } from 'vitest'
import { createLaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import {
  Config,
  DEFAULT_ACCESS_TOKEN_ENV,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  DEFAULT_MODELS,
  resolveAdapterOptions,
  vertexOpenAiRoot,
  wireModelId,
} from '../src/index.ts'

describe('Config schema', () => {
  it('fills every default so an empty entry resolves', () => {
    const value = new Config({})
    expect(value.auth).toBe('adc')
    expect(value.accessTokenEnv).toBe(DEFAULT_ACCESS_TOKEN_ENV)
    expect(value.scopes).toEqual(['https://www.googleapis.com/auth/cloud-platform'])
    expect(value.location).toBe('global')
    expect(value.publisher).toBe('google')
    expect(value.reasoningEffort).toBe('high')
    expect(value.maxTokens).toBe(DEFAULT_MAX_TOKENS)
    expect(value.defaultContextWindow).toBe(DEFAULT_CONTEXT_WINDOW)
    expect(value.models).toEqual(DEFAULT_MODELS)
    expect(value.project).toBeUndefined()
    expect(value.baseURL).toBeUndefined()
  })

  it('refuses an effort Vertex has no spelling for', () => {
    expect(() => new Config({ reasoningEffort: 'max' as never })).toThrow()
  })
})

describe('resolveAdapterOptions', () => {
  it('resolves the shipped defaults from an empty entry', () => {
    const options = resolveAdapterOptions({})
    expect(options).toMatchObject({
      auth: 'adc',
      accessTokenEnv: DEFAULT_ACCESS_TOKEN_ENV,
      location: 'global',
      publisher: 'google',
      reasoningEffort: 'high',
      maxTokens: DEFAULT_MAX_TOKENS,
      defaultContextWindow: DEFAULT_CONTEXT_WINDOW,
      project: undefined,
      baseURL: undefined,
    })
    expect(options.models).toEqual(DEFAULT_MODELS)
    expect(options.retryPolicy.mode).toBe('normal')
  })

  it('reads the project from the launch environment in documented order', () => {
    const both = createLaunchEnvironmentSnapshot([{ source: 'process', values: { GOOGLE_CLOUD_PROJECT: 'from-google', GCLOUD_PROJECT: 'from-gcloud' } }])
    expect(resolveAdapterOptions({}, both).project).toBe('from-google')
    const gcloud = createLaunchEnvironmentSnapshot([{ source: 'process', values: { GCLOUD_PROJECT: 'from-gcloud', GOOGLE_CLOUD_PROJECT: '' } }])
    expect(resolveAdapterOptions({}, gcloud).project).toBe('from-gcloud')
    expect(resolveAdapterOptions({ project: 'explicit' }, both).project).toBe('explicit')
  })

  it('detaches and validates the catalog', () => {
    const models = [{ id: 'gemini-3.8-pro', name: 'Pro', description: 'd', contextWindow: 10, maxTokens: 5 }]
    const resolved = resolveAdapterOptions({ models })
    expect(resolved.models).toEqual(models)
    expect(resolved.models).not.toBe(models)
    expect(resolveAdapterOptions({ models: [{ id: 'bare' }] }).models).toEqual([{ id: 'bare' }])
    expect(() => resolveAdapterOptions({ models: [{ id: '' }] })).toThrow('ids must be non-empty')
    expect(() => resolveAdapterOptions({ models: [{ id: 'm', name: '' }] })).toThrow('empty name')
    expect(() => resolveAdapterOptions({ models: [{ id: 'm', contextWindow: 0 }] })).toThrow('contextWindow')
    expect(() => resolveAdapterOptions({ models: [{ id: 'm', maxTokens: 1.5 }] })).toThrow('maxTokens')
    expect(() => resolveAdapterOptions({ models: [{ id: 'm' }, { id: 'm' }] })).toThrow('duplicate')
  })

  it('rejects beyond-schema values a settings snapshot can carry', () => {
    expect(() => resolveAdapterOptions({ auth: 'oauth' as never })).toThrow('auth must be')
    expect(() => resolveAdapterOptions({ reasoningEffort: 'max' as never })).toThrow('reasoningEffort must be')
    expect(() => resolveAdapterOptions({ scopes: [] })).toThrow('scopes')
    expect(() => resolveAdapterOptions({ scopes: [''] })).toThrow('scopes')
    expect(() => resolveAdapterOptions({ defaultContextWindow: 0 })).toThrow('defaultContextWindow')
    expect(() => resolveAdapterOptions({ maxTokens: -1 })).toThrow('maxTokens')
    expect(() => resolveAdapterOptions({ streamIdleTimeoutMs: 0 })).toThrow('streamIdleTimeoutMs')
    expect(() => resolveAdapterOptions({ streamIdleTimeoutMs: MAX_TIMER_DELAY_MS + 1 })).toThrow('streamIdleTimeoutMs')
    expect(() => resolveAdapterOptions({ project: '' })).toThrow('project must be')
    expect(() => resolveAdapterOptions({ location: '' })).toThrow('location must be')
    expect(() => resolveAdapterOptions({ publisher: '' })).toThrow('publisher must be')
    expect(() => resolveAdapterOptions({ accessTokenEnv: '' })).toThrow('accessTokenEnv must be')
    expect(() => resolveAdapterOptions({ baseURL: '' })).toThrow('baseURL must be')
    expect(() => resolveAdapterOptions({ baseURL: 'ftp://gateway' })).toThrow('HTTP(S) root')
    expect(() => resolveAdapterOptions({ baseURL: 'https://user:pw@gateway' })).toThrow('HTTP(S) root')
    expect(() => resolveAdapterOptions({ baseURL: 'https://gateway/v1?x=1' })).toThrow('HTTP(S) root')
  })

  it('keeps an explicit gateway root and a regional location', () => {
    const options = resolveAdapterOptions({
      baseURL: 'https://gateway.example/openai',
      location: 'europe-west1',
      auth: 'access-token',
      accessTokenEnv: 'VERTEX_TOKEN',
      scopes: ['scope-a'],
      streamIdleTimeoutMs: 1234,
      retryPolicy: { mode: 'normal', maxRetries: 0 },
    })
    expect(options).toMatchObject({
      baseURL: 'https://gateway.example/openai',
      location: 'europe-west1',
      auth: 'access-token',
      accessTokenEnv: 'VERTEX_TOKEN',
      scopes: ['scope-a'],
      streamIdleTimeoutMs: 1234,
    })
    expect(options.retryPolicy).toMatchObject({ mode: 'normal', maxRetries: 0 })
  })
})

describe('endpoint and model addressing', () => {
  it('serves the global location from the bare host and regions from their prefix', () => {
    expect(vertexOpenAiRoot('my-proj', 'global'))
      .toBe('https://aiplatform.googleapis.com/v1/projects/my-proj/locations/global/endpoints/openapi')
    expect(vertexOpenAiRoot('my proj', 'europe-west1'))
      .toBe('https://europe-west1-aiplatform.googleapis.com/v1/projects/my%20proj/locations/europe-west1/endpoints/openapi')
  })

  it('qualifies bare model ids with the publisher and passes qualified ids through', () => {
    expect(wireModelId('gemini-3.8-flash', 'google')).toBe('google/gemini-3.8-flash')
    expect(wireModelId('meta/llama', 'google')).toBe('meta/llama')
  })
})
