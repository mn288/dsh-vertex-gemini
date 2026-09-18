/**
 * Google OAuth bearer minting through Application Default Credentials. The
 * `google-auth-library` client owns discovery (gcloud user login, service
 * account file, workload identity, metadata server), token caching, and
 * refresh; this module owns the harness failure vocabulary around it. The
 * library loads on the first bearer request, so a composition whose Vertex
 * route stays dormant, or uses a stored access token, never pays for it.
 *
 * @module dsh-llm-vertex/auth
 */

import type { AuthClient } from 'google-auth-library'
import { LlmError } from '@deepseek-ai/dsh-llm'

/** Per-request bearers minted from one credential discovery. */
export interface GoogleCredentialSource {
  /**
   * Return a currently valid bearer, refreshing through the credential client
   * when the cached one is near expiry. Throws `LlmError` `MISSING_CREDENTIAL`
   * when no credentials can be discovered and `AUTH` when discovered
   * credentials cannot produce a token (revoked or expired login).
   */
  accessToken(): Promise<string>
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Discover Application Default Credentials once and mint bearers from them.
 * Discovery is retried on the next call after a failure, so a login completed
 * while the harness runs is picked up without a restart.
 * @param scopes - OAuth scopes requested for every token.
 * @returns a source bound to one credential discovery.
 */
export function adcCredentialSource(scopes: readonly string[]): GoogleCredentialSource {
  let client: Promise<AuthClient> | undefined
  const resolveClient = (): Promise<AuthClient> => {
    client ??= import('google-auth-library')
      .then(({ GoogleAuth }) => new GoogleAuth({ scopes: [...scopes] }).getClient())
      .catch((error: unknown) => {
        client = undefined
        throw new LlmError(
          'llm-vertex: no Google Application Default Credentials; run "gcloud auth application-default login"'
          + ` or point GOOGLE_APPLICATION_CREDENTIALS at a service-account key (${describe(error)})`,
          'MISSING_CREDENTIAL',
          { cause: error },
        )
      })
    return client
  }
  return {
    async accessToken(): Promise<string> {
      const resolved = await resolveClient()
      let token: string | null | undefined
      try {
        token = (await resolved.getAccessToken()).token
      } catch (error: unknown) {
        throw new LlmError(
          `llm-vertex: Google credentials could not produce an access token (${describe(error)})`,
          'AUTH',
          { cause: error },
        )
      }
      if (typeof token !== 'string' || token.length === 0) {
        throw new LlmError('llm-vertex: Google credentials returned an empty access token', 'AUTH')
      }
      return token
    },
  }
}
