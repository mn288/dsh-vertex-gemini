/** Register Google Vertex AI Gemini with request-local settings, Application Default Credentials, and thought-signature replay. */
import type { Context } from '@deepseek-ai/cordis'
import { assertUsableApiKey, LlmError } from '@deepseek-ai/dsh-llm'
import type { AdapterRegistrationHandle } from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-settings'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import { deepEqualJson } from '@deepseek-ai/dsh-util-values'
import { VertexAdapter } from './adapter.ts'
import { adcCredentialSource } from './auth.ts'
import type { GoogleCredentialSource } from './auth.ts'
import { Config, resolveAdapterOptions } from './config.ts'
import type { VertexConnectionOptions } from './config.ts'

export { Config, resolveAdapterOptions, vertexOpenAiRoot, wireModelId } from './config.ts'
export {
  DEFAULT_ACCESS_TOKEN_ENV,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_LOCATION,
  DEFAULT_MAX_TOKENS,
  DEFAULT_MODELS,
  DEFAULT_PUBLISHER,
  DEFAULT_REASONING_EFFORT,
  DEFAULT_SCOPES,
  DEFAULT_STREAM_IDLE_TIMEOUT_MS,
  PROJECT_ENV_NAMES,
} from './config.ts'
export type { VertexAuthMode, VertexCatalogModel, VertexConnectionOptions, VertexReasoningEffort } from './config.ts'
export { VertexAdapter, httpErrorCode } from './adapter.ts'
export type { VertexAdapterOptions } from './adapter.ts'
export { adcCredentialSource } from './auth.ts'
export type { GoogleCredentialSource } from './auth.ts'
export { readReplay, replayState, REPLAY_KIND, REPLAY_VERSION, THOUGHT_SIGNATURE_SKIP_MARKER } from './replay.ts'
export type { ReplayBlock } from './replay.ts'
export { REASONING_EFFORTS } from './model-info.ts'
export type * from './types.ts'

export const name = 'llm-vertex'
export const inject = ['llm']

const NS = 'llm-vertex'
/**
 * The one provider route this plugin registers. `google-vertex` is already a
 * pi-ai catalog id that `llm-pi-ai` lists in the configurable directory, so
 * this route takes a distinct key to coexist with that adapter.
 */
export const PROVIDER = 'vertex-ai'

/**
 * The routes one configuration generation serves: the route is live once an
 * endpoint can be addressed — a project, or an explicit gateway root — and
 * dormant otherwise, so a machine that never configures Vertex sees a
 * configurable provider in the directory and no live route in the picker.
 * @param connection - validated connection facts.
 * @returns the complete route set for this generation.
 */
export function routesFor(connection: VertexConnectionOptions): string[] {
  return connection.baseURL !== undefined || connection.project !== undefined ? [PROVIDER] : []
}

export function apply(ctx: Context, config: Config): void {
  let current: () => Config = () => config
  let lastRaw: Config | undefined
  let lastGood: VertexConnectionOptions | undefined
  /* jscpd:ignore-start -- mirrors llm-deepseek last-good settings snapshot; siblings stay symmetric until the protocol is shared */
  const options = (): VertexConnectionOptions => {
    const raw = current()
    if (raw === lastRaw && lastGood !== undefined) return lastGood
    try {
      const next = resolveAdapterOptions(raw, launchEnvironmentOf(ctx))
      lastRaw = raw
      lastGood = next
      return next
    } catch (error) {
      // Static composition resolves before anything registers, so this branch
      // only sees a live settings snapshot failing a beyond-schema bound:
      // keep serving the last good facts and say so once per bad snapshot.
      if (lastGood === undefined) throw error
      lastRaw = raw
      ctx.logger.error('llm-vertex: keeping the last good configuration after an invalid settings section')
      ctx.logger.error(error)
      return lastGood
    }
  }
  options()

  /* jscpd:ignore-end */
  // One credential discovery per scope set: the client caches and refreshes
  // its token, so re-discovering per request would re-read the login files
  // and re-mint a bearer every call.
  let credentialSource: { scopes: string; source: GoogleCredentialSource } | undefined
  const credentialsFor = (connection: VertexConnectionOptions): GoogleCredentialSource => {
    const scopes = JSON.stringify(connection.scopes)
    if (credentialSource?.scopes !== scopes) credentialSource = { scopes, source: adcCredentialSource(connection.scopes) }
    return credentialSource.source
  }

  const resolveBearer = async (connection: VertexConnectionOptions): Promise<string> => {
    if (connection.auth === 'adc') return credentialsFor(connection).accessToken()
    // Every credential fact comes from the caller's snapshot, so a rejected
    // settings generation cannot leak its token onto the previous endpoint.
    const ref = connection.accessTokenEnv
    const credentials = ctx.get('credentials')
    if (credentials !== undefined) {
      const hit = await credentials.resolve(ref)
      if (hit !== undefined) return assertUsableApiKey(hit.value, 'llm-vertex', ref)
    } else {
      // Without the seam there is no managed store to rank against, so the
      // environment is the whole credential plane.
      const ambient = launchEnvironmentOf(ctx).get(ref)
      if (ambient !== undefined && ambient.value.length > 0) {
        return assertUsableApiKey(ambient.value, 'llm-vertex', ref)
      }
    }
    throw new LlmError(
      `llm-vertex: no access token for provider route "${PROVIDER}"; store ${ref} through the credentials`
      + ` service (the web Models page writes it), export ${ref} in the launching environment, or switch to auth: adc`,
      'MISSING_CREDENTIAL',
    )
  }

  const adapter = new VertexAdapter({
    options,
    resolveBearer,
    onReplayDegrade: ({ provider, model, reason }) => {
      ctx.logger.warn(`llm-vertex: unusable thought-signature replay state on assistant history for route "${provider}/${model}"; sending the skip marker (${reason})`)
    },
  })
  ctx.llm.registerConfigurableProviders([
    { provider: PROVIDER, displayName: 'Google Vertex AI', settingsNs: NS, settingsPath: [] },
  ])

  // The registry captures the route set and the retry policy at registration;
  // they are the two facts per-request resolution cannot refresh. An initial
  // registration must name at least one route, so the handle is created on
  // the first generation that addresses an endpoint and `replace` carries
  // every later change in one synchronous registry section, so no request
  // observes a gap. Route effects bind to this apply fiber via the stable
  // `ctx` reference, even when a swap runs inside the scoped settings
  // callback below.
  let registration: AdapterRegistrationHandle | undefined
  let registered: { routes: string[]; policy: VertexConnectionOptions['retryPolicy'] } | undefined
  const ensureRegistrationFacts = (): void => {
    const connection = options()
    const routes = routesFor(connection)
    const policy = connection.retryPolicy
    if (registered !== undefined && deepEqualJson(routes, registered.routes) && deepEqualJson(policy, registered.policy)) return
    if (registration === undefined) {
      if (routes.length === 0) return
      registration = ctx.llm.registerAdapter(routes, adapter)
    } else {
      registration.replace(routes)
    }
    registered = { routes, policy }
  }
  ensureRegistrationFacts()

  ctx.inject(['settings'], (settingsCtx) => {
    settingsCtx.settings.installSection(ctx, NS, Config, config, {
      setSource: (source) => {
        current = source
      },
      onChange: ensureRegistrationFacts,
    })
  })
}
