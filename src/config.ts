/** Plugin configuration and complete request-local resolution for Vertex AI Gemini. */
import z from '@deepseek-ai/schemastery'
import { resolveRetryPolicy, RetryPolicySchema } from '@deepseek-ai/dsh-llm'
import type { ResolvedRetryPolicy, RetryPolicyConfig } from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import type { CredentialRef } from '@deepseek-ai/dsh-credentials'
import type { LaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'

/** Default access-token credential reference for `auth: access-token`. */
export const DEFAULT_ACCESS_TOKEN_ENV = 'GOOGLE_OAUTH_ACCESS_TOKEN'
/** OAuth scope every Vertex AI request needs. */
export const DEFAULT_SCOPES = ['https://www.googleapis.com/auth/cloud-platform'] as const
/** Vertex location serving the global Gemini endpoints. */
export const DEFAULT_LOCATION = 'global'
/** Model publisher prefixed to unqualified model ids. */
export const DEFAULT_PUBLISHER = 'google'
/** Default per-request output cap: Gemini 3.8 Flash's published maximum. */
export const DEFAULT_MAX_TOKENS = 65_536
/** Default context capacity: Gemini 3.8's published window. */
export const DEFAULT_CONTEXT_WINDOW = 1_048_576
/** Default maximum provider idle time while one stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
/** Default reasoning effort when configuration names none. */
export const DEFAULT_REASONING_EFFORT = 'high'

/** Environment variables naming the project, honored in this order when configuration names none. */
export const PROJECT_ENV_NAMES = ['GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT'] as const

/** How the adapter obtains its Google OAuth bearer. */
export type VertexAuthMode = 'adc' | 'access-token'

/** Gemini thinking levels selectable through this adapter. */
export type VertexReasoningEffort = 'low' | 'medium' | 'high'

/** One advisory model entry served by the `vertex-ai` route. */
export interface VertexCatalogModel {
  /** Model id; a bare id is prefixed with the configured publisher on the wire. */
  id: string
  /** Selector label; defaults to {@link id}. */
  name?: string
  /** Optional selector detail for deployments with similar model variants. */
  description?: string
  /** Known combined request/response context capacity. */
  contextWindow?: number
  /** Per-request output cap for this model; omission falls back to the profile's `maxTokens`. */
  maxTokens?: number
}

/** Shipped catalog: the Gemini 3.8 model this adapter was verified against. */
export const DEFAULT_MODELS: readonly VertexCatalogModel[] = [
  {
    id: 'gemini-3.8-flash',
    name: 'Gemini 3.8 Flash',
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: DEFAULT_MAX_TOKENS,
  },
]

/**
 * Plugin config, validated by the same-named schemastery schema and doubling
 * as the `llm-vertex` settings-section shape. Every field is optional: with
 * nothing configured the plugin loads and lists a configurable provider, and
 * the route goes live once a project (configuration or environment) or an
 * explicit `baseURL` addresses an endpoint. A missing Google login fails the
 * request (`MISSING_CREDENTIAL`), not the plugin load.
 */
export interface Config {
  /**
   * `adc` mints bearers from Application Default Credentials; `access-token`
   * reads a stored bearer through {@link Config.accessTokenEnv}.
   */
  auth?: VertexAuthMode
  /** Credential reference holding a Google OAuth access token; used only with `auth: access-token`. */
  accessTokenEnv?: string
  /** OAuth scopes requested from Application Default Credentials. */
  scopes?: string[]
  /** Google Cloud project id; falls back to `$GOOGLE_CLOUD_PROJECT`, then `$GCLOUD_PROJECT`. */
  project?: string
  /** Vertex AI location (default `global`). */
  location?: string
  /** Publisher prefixed to bare model ids on the wire (default `google`). */
  publisher?: string
  /** Complete OpenAI-compatible root overriding the computed Vertex endpoint; project and location are then not needed for routing. */
  baseURL?: string
  /** Default thinking level (default `high`). */
  reasoningEffort?: VertexReasoningEffort
  /** Default per-request output cap; a model's own cap and explicit request values win. */
  maxTokens?: number
  /** Positive context capacity used when the selected model has no exact value. */
  defaultContextWindow?: number
  /** Advisory models shown by discovery consumers; defaults to Gemini 3.8 Flash. */
  models?: VertexCatalogModel[]
  /** Maximum provider idle time while one stream read is outstanding (default five minutes). */
  streamIdleTimeoutMs?: number
  /** Provider-owned model-request retry policy; omission uses normal mode with five retries. */
  retryPolicy?: RetryPolicyConfig
}

/* jscpd:ignore-start -- mirrors llm-deepseek config schema; siblings stay symmetric until the protocol is shared */
const catalogModel: z<VertexCatalogModel> = z.object({
  id: z.string().required(),
  name: z.string(),
  description: z.string(),
  contextWindow: z.number().step(1).min(1),
  maxTokens: z.number().step(1).min(1),
})

/* jscpd:ignore-end */
export const Config: z<Config> = z.object({
  auth: z.union(['adc', 'access-token']).default('adc'),
  accessTokenEnv: z.string().role('credential-ref').default(DEFAULT_ACCESS_TOKEN_ENV),
  scopes: z.array(z.string()).default([...DEFAULT_SCOPES]),
  project: z.string(),
  location: z.string().default(DEFAULT_LOCATION),
  publisher: z.string().default(DEFAULT_PUBLISHER),
  baseURL: z.string(),
  reasoningEffort: z.union(['low', 'medium', 'high']).default(DEFAULT_REASONING_EFFORT),
  maxTokens: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER).default(DEFAULT_MAX_TOKENS),
  defaultContextWindow: z.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
  models: z.array(catalogModel).default([...DEFAULT_MODELS]),
  streamIdleTimeoutMs: z.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  retryPolicy: RetryPolicySchema,
})

/**
 * Validated connection facts for one operation. Endpoint, credential mode, and
 * catalog are one value on purpose: a settings snapshot the resolver rejects
 * keeps the whole previous generation, so a request can never pair a stale
 * endpoint with a newer credential.
 */
export interface VertexConnectionOptions {
  /** How the bearer is obtained for this generation. */
  auth: VertexAuthMode
  /** Credential reference for `access-token` mode, resolved per request. */
  accessTokenEnv: CredentialRef
  /** OAuth scopes for `adc` mode. */
  scopes: readonly string[]
  /** Project from configuration or the launch environment; absent keeps the route dormant unless `baseURL` is set. */
  project: string | undefined
  /** Vertex location. */
  location: string
  /** Publisher prefixed to bare model ids. */
  publisher: string
  /** Explicit OpenAI-compatible root, when configured. */
  baseURL: string | undefined
  /** Default thinking level. */
  reasoningEffort: VertexReasoningEffort
  /** Default per-request output cap; explicit request values win. */
  maxTokens: number
  /** Positive context capacity used when the selected model has no exact value. */
  defaultContextWindow: number
  /** Advisory models exposed to discovery consumers; requests remain unrestricted. */
  models: readonly VertexCatalogModel[]
  /** Maximum provider idle time while one stream read is outstanding. */
  streamIdleTimeoutMs: number
  /** Provider-owned model-request retry policy, already resolved. */
  retryPolicy: ResolvedRetryPolicy
}

/* jscpd:ignore-start -- mirrors llm-deepseek resolve step; siblings stay symmetric until the protocol is shared */
/** Resolve, validate, and detach the advisory model catalog. */
function resolveModels(models: readonly VertexCatalogModel[] | undefined): VertexCatalogModel[] {
  const seen = new Set<string>()
  return (models ?? DEFAULT_MODELS).map((model) => {
    if (model.id.length === 0) throw new Error('llm-vertex: catalog model ids must be non-empty')
    if (model.name !== undefined && model.name.length === 0) {
      throw new Error(`llm-vertex: catalog model "${model.id}" has an empty name`)
    }
    if (model.contextWindow !== undefined
      && (!Number.isInteger(model.contextWindow) || model.contextWindow <= 0)) {
      throw new Error(`llm-vertex: catalog model "${model.id}" contextWindow must be a positive integer`)
    }
    if (model.maxTokens !== undefined
      && (!Number.isInteger(model.maxTokens) || model.maxTokens <= 0)) {
      throw new Error(`llm-vertex: catalog model "${model.id}" maxTokens must be a positive integer`)
    }
    if (seen.has(model.id)) throw new Error(`llm-vertex: duplicate catalog model "${model.id}"`)
    seen.add(model.id)
    return {
      id: model.id,
      ...model.name === undefined ? {} : { name: model.name },
      ...model.description === undefined ? {} : { description: model.description },
      ...model.contextWindow === undefined ? {} : { contextWindow: model.contextWindow },
      ...model.maxTokens === undefined ? {} : { maxTokens: model.maxTokens },
    }
  })
}
/* jscpd:ignore-end */

/** Reject a non-empty-string bound at both the composition entry and a live settings snapshot. */
function requireNonEmpty(value: string | undefined, field: string): string | undefined {
  if (value !== undefined && value.length === 0) throw new Error(`llm-vertex: ${field} must be a non-empty string`)
  return value
}

/**
 * The one explicit resolve step from raw config to validated connection
 * facts. Programmatic construction may bypass Schemastery normalization, so
 * every default and bound is re-judged here — for the composition entry at
 * load (fail loud) and for each settings snapshot at its first use.
 * @param config - raw plugin config or resolved settings snapshot.
 * @param environment - this run's environment layers, or `undefined` outside
 *   the product CLI; every layer may name the project.
 * @returns validated connection facts.
 */
export function resolveAdapterOptions(config: Config, environment?: LaunchEnvironmentSnapshot): VertexConnectionOptions {
  const auth: string = config.auth ?? 'adc'
  if (auth !== 'adc' && auth !== 'access-token') {
    throw new Error('llm-vertex: auth must be adc or access-token')
  }
  const reasoningEffort: string = config.reasoningEffort ?? DEFAULT_REASONING_EFFORT
  if (reasoningEffort !== 'low' && reasoningEffort !== 'medium' && reasoningEffort !== 'high') {
    throw new Error('llm-vertex: reasoningEffort must be low, medium, or high')
  }
  const scopes = config.scopes ?? [...DEFAULT_SCOPES]
  if (scopes.length === 0 || scopes.some(scope => scope.length === 0)) {
    throw new Error('llm-vertex: scopes must list at least one non-empty scope')
  }
  if (config.defaultContextWindow !== undefined
    && (!Number.isInteger(config.defaultContextWindow) || config.defaultContextWindow <= 0)) {
    throw new Error('llm-vertex: defaultContextWindow must be a positive integer')
  }
  if (config.maxTokens !== undefined
    && (!Number.isSafeInteger(config.maxTokens) || config.maxTokens <= 0)) {
    throw new Error('llm-vertex: maxTokens must be a positive safe integer')
  }
  const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
  if (!Number.isFinite(streamIdleTimeoutMs)
    || streamIdleTimeoutMs <= 0
    || streamIdleTimeoutMs > MAX_TIMER_DELAY_MS) {
    throw new Error(
      `llm-vertex: streamIdleTimeoutMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`,
    )
  }
  const baseURL = requireNonEmpty(config.baseURL, 'baseURL')
  if (baseURL !== undefined) {
    const parsed = new URL(baseURL)
    if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash) {
      throw new Error('llm-vertex: baseURL must be an HTTP(S) root without credentials, query, or fragment')
    }
  }
  const project = requireNonEmpty(config.project, 'project')
    ?? PROJECT_ENV_NAMES.map(name => environment?.get(name)?.value).find(value => value !== undefined && value.length > 0)
  return {
    auth,
    accessTokenEnv: credentialRef(requireNonEmpty(config.accessTokenEnv, 'accessTokenEnv') ?? DEFAULT_ACCESS_TOKEN_ENV),
    scopes: [...scopes],
    project,
    location: requireNonEmpty(config.location, 'location') ?? DEFAULT_LOCATION,
    publisher: requireNonEmpty(config.publisher, 'publisher') ?? DEFAULT_PUBLISHER,
    baseURL,
    reasoningEffort,
    maxTokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
    defaultContextWindow: config.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW,
    models: resolveModels(config.models),
    streamIdleTimeoutMs,
    retryPolicy: resolveRetryPolicy(config.retryPolicy, 'llm-vertex: retryPolicy'),
  }
}

/**
 * The OpenAI-compatible root for one project and location. The `global`
 * location is served from the un-prefixed host; every regional location is
 * served from `{location}-aiplatform.googleapis.com`.
 * @param project - Google Cloud project id.
 * @param location - Vertex AI location.
 * @returns the root that `/chat/completions` is appended to.
 */
export function vertexOpenAiRoot(project: string, location: string): string {
  const host = location === 'global' ? 'aiplatform.googleapis.com' : `${location}-aiplatform.googleapis.com`
  return `https://${host}/v1/projects/${encodeURIComponent(project)}/locations/${encodeURIComponent(location)}/endpoints/openapi`
}

/**
 * Qualify a model id for the wire: Vertex addresses publisher models as
 * `{publisher}/{model}`, and an id that already names its publisher passes through.
 * @param model - harness model id.
 * @param publisher - configured publisher for bare ids.
 * @returns the wire model id.
 */
export function wireModelId(model: string, publisher: string): string {
  return model.includes('/') ? model : `${publisher}/${model}`
}
