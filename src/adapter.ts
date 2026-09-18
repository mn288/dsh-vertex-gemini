/**
 * `VertexAdapter`: fetch + SSE against the Vertex AI OpenAI-compatible
 * chat-completions endpoint for Gemini, emitting harness StreamChunks. The
 * adapter is transport-only: connection facts arrive through a thunk resolved
 * once per operation, and the Google bearer and project through per-request
 * resolvers, so the registering plugin owns validation, layering, and
 * credential policy.
 *
 * @module dsh-llm-vertex/adapter
 */

import {
  attributionHeaders,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  isContextWindowExceededError,
  isQuotaExceededError,
  LlmAdapter,
  LlmError,
  QUOTA_EXCEEDED_CODE,
} from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  PreparedAdapterCall,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { vertexOpenAiRoot } from './config.ts'
import type { VertexConnectionOptions } from './config.ts'
import { catalogModelInfo, modelInfo } from './model-info.ts'
import { serializeRequest } from './serialize.ts'
import { parseSse } from './sse.ts'
import { translate } from './translate.ts'
import type { WireError, WireErrorRecord } from './types.ts'

const STREAM_IDLE_TIMEOUT_CODE = 'LLM_STREAM_IDLE_TIMEOUT'

/** Constructor options for {@link VertexAdapter}: the operation-local resolution hooks the plugin owns. */
export interface VertexAdapterOptions {
  /** Current validated connection facts; called once per operation. */
  options: () => VertexConnectionOptions
  /**
   * Resolve the Google OAuth bearer for the connection facts of one request.
   * The snapshot is passed in — never re-read — so the credential can only
   * come from the same resolution as the endpoint it is sent to. Throws
   * `LlmError` `MISSING_CREDENTIAL` or `AUTH`.
   */
  resolveBearer: (connection: VertexConnectionOptions) => Promise<string>
  /** Report unusable replay metadata without exposing content or signatures. */
  onReplayDegrade?: (detail: { provider: string; model: string; reason: string }) => void
}

/** Normalize the two error-body layouts Vertex and gateways send into one record. */
function errorRecord(body: WireError): WireErrorRecord['error'] {
  return (Array.isArray(body) ? body[0] : body)?.error
}

/**
 * Map an HTTP status to a stable LlmError code.
 * @param status - status of a non-2xx provider response.
 * @param error - parsed Google error record, when available.
 * @returns the normalized harness error code.
 */
export function httpErrorCode(status: number, error?: WireErrorRecord['error']): string {
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 404) return 'NOT_FOUND'
  if (status === 413) return 'INVALID_REQUEST'
  const detail = [error?.status, error?.message].filter(Boolean).join(' ')
  if (status === 429) return isQuotaExceededError(detail) ? QUOTA_EXCEEDED_CODE : 'RATE_LIMIT'
  if (status === 400) {
    return isContextWindowExceededError(detail) ? CONTEXT_WINDOW_EXCEEDED_CODE : 'INVALID_REQUEST'
  }
  if (status >= 500) return 'SERVER'
  return `HTTP_${status}`
}

/**
 * The OpenAI-compatible root one connection generation addresses: an explicit
 * gateway root, else the Vertex endpoint of its project and location. Throws
 * `LlmError` `MISSING_PROJECT` when neither is configured; the plugin keeps
 * such a generation's route dormant, so this guards direct adapter use.
 * @param connection - validated connection facts.
 * @returns the root that `/chat/completions` is appended to.
 */
export function endpointRoot(connection: VertexConnectionOptions): string {
  if (connection.baseURL !== undefined) return connection.baseURL
  if (connection.project !== undefined) return vertexOpenAiRoot(connection.project, connection.location)
  throw new LlmError(
    'llm-vertex: no Google Cloud project; set "project" in the llm-vertex settings section,'
    + ' export GOOGLE_CLOUD_PROJECT, or configure an explicit baseURL',
    'MISSING_PROJECT',
  )
}

/**
 * Settle with `pending` unless `signal` aborts first, then reject with the
 * signal's reason. The credential client takes no signal, so an abandoned
 * mint settles on its own and its result is dropped.
 */
async function untilAborted<T>(pending: Promise<T>, signal: AbortSignal): Promise<T> {
  signal.throwIfAborted()
  const aborted = Promise.withResolvers<never>()
  const onAbort = (): void => { aborted.reject(signal.reason) }
  signal.addEventListener('abort', onAbort, { once: true })
  try {
    return await Promise.race([pending, aborted.promise])
  } finally {
    signal.removeEventListener('abort', onAbort)
  }
}

function providerRetryAfterMs(value: string | null): number | undefined {
  if (value === null) return undefined
  if (/^\d+$/.test(value)) {
    const delay = Number(value) * 1_000
    return delay > 0 ? delay : undefined
  }
  const delay = Date.parse(value) - Date.now()
  return Number.isFinite(delay) && delay > 0 ? delay : undefined
}

/**
 * Vertex AI Gemini over the OpenAI-compatible endpoint. One instance serves
 * every model id under its route; the harness model id becomes the wire id
 * through the configured publisher prefix.
 *
 * One stable signal reaches the bearer mint, the initial fetch, and body
 * reads. Caller aborts map to `ABORTED`; the configured per-read idle watchdog
 * maps to `TIMEOUT`.
 */
export class VertexAdapter extends LlmAdapter {
  constructor(private readonly config: VertexAdapterOptions) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Google Vertex AI' }
  }

  override providerRetryPolicy(_provider: string): ResolvedRetryPolicy {
    return this.config.options().retryPolicy
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve(this.config.options().models.map(model => catalogModelInfo(provider, model)))
  }

  /* jscpd:ignore-start -- mirrors llm-deepseek adapter surface and stream loop; siblings stay symmetric until the protocol is shared */
  override resolveModel(
    provider: string,
    model: string,
    _signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    return Promise.resolve(modelInfo(this.config.options(), provider, model))
  }

  override prepareCall(provider: string, model: string, _signal?: AbortSignal): Promise<PreparedAdapterCall> {
    const connection = this.config.options()
    return Promise.resolve({
      model: modelInfo(connection, provider, model),
      stream: options => this.streamWithConnection(options, connection),
    })
  }

  stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    return this.streamWithConnection(options, this.config.options())
  }

  private async * streamWithConnection(
    options: GenerateOptions,
    connection: VertexConnectionOptions,
  ): AsyncIterable<StreamChunk> {
    // One resolution per stream call: connection facts and endpoint freeze
    // here, and the bearer is minted from the same facts as the request's
    // first step, so an in-flight stream never observes a configuration
    // change and the next call re-resolves.
    const root = endpointRoot(connection)
    const consumer = new AbortController()
    const upstream = options.signal === undefined
      ? consumer.signal
      : AbortSignal.any([options.signal, consumer.signal])
    using watchdog = idleWatchdog(upstream, connection.streamIdleTimeoutMs, STREAM_IDLE_TIMEOUT_CODE)
    const iterator = this.request(options, watchdog.signal, connection, root, () => { watchdog.pulse() })[Symbol.asyncIterator]()
    let exhausted = false
    try {
      while (true) {
        const result = await watchdog.next(iterator)
        if (result.done) {
          exhausted = true
          return
        }
        yield result.value
      }
    } catch (error: unknown) {
      if (timeoutOf(watchdog.signal, STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
        throw new LlmError(
          `Vertex stream idle timeout after ${connection.streamIdleTimeoutMs}ms`,
          'TIMEOUT',
          { cause: error },
        )
      }
      if (options.signal?.aborted) {
        throw new LlmError('Vertex request aborted by caller', 'ABORTED', { cause: error })
      }
      if (error instanceof LlmError) throw error
      throw new LlmError(`Vertex API stream from ${root} failed`, 'TRANSPORT', { cause: error })
    } finally {
      consumer.abort('Vertex stream consumer stopped')
      if (!exhausted && iterator.return !== undefined) {
        try {
          await iterator.return()
        } catch (_abortedTransportTeardown) {
          // The consumer controller already owns termination; a return-time abort cannot add a second outcome.
        }
      }
    }
  }

  /* jscpd:ignore-end */
  private async * request(
    options: GenerateOptions,
    signal: AbortSignal,
    connection: VertexConnectionOptions,
    root: string,
    onActivity: () => void,
  ): AsyncIterable<StreamChunk> {
    // Minted under the request signal: a caller abort or the idle budget ends
    // a stalled credential discovery or token refresh like a stalled read.
    const bearer = await untilAborted(this.config.resolveBearer(connection), signal)
    const body = serializeRequest(options, {
      connection,
      onReplayDegrade: reason => this.config.onReplayDegrade?.({ provider: options.provider, model: options.model, reason }),
    })
    const headers = {
      'authorization': `Bearer ${bearer}`,
      'content-type': 'application/json',
      'accept': 'text/event-stream',
      ...attributionHeaders(),
    }
    let response: Response
    try {
      response = await fetch(`${root}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal,
      })
    } catch (error: unknown) {
      if (signal.aborted) throw error
      throw new LlmError(`Vertex API request to ${root} failed`, 'TRANSPORT', { cause: error })
    }

    if (!response.ok) {
      let message = `Vertex API error (HTTP ${response.status})`
      let providerError: WireErrorRecord['error']
      const rawResponse = await response.text()
      try {
        providerError = errorRecord(JSON.parse(rawResponse) as WireError)
        if (providerError?.message) message = providerError.message
      } catch {
        // The HTTP status remains authoritative when a gateway returns malformed JSON.
      }
      const delay = providerRetryAfterMs(response.headers.get('retry-after'))
      throw new LlmError(message, httpErrorCode(response.status, providerError), {
        cause: new Error(rawResponse.length > 0 ? rawResponse : `Vertex HTTP ${response.status}`),
        status: response.status,
        ...delay === undefined ? {} : { providerRetryAfterMs: delay },
      })
    }
    if (!response.body) {
      throw new LlmError('Vertex API returned no response body', 'EMPTY_RESPONSE')
    }

    yield* translate(parseSse(response.body, onActivity), options.model)
  }
}
