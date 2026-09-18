import { afterEach, describe, expect, it, vi } from 'vitest'
import { LlmError } from '@deepseek-ai/dsh-llm'

/** The one credential client the mocked library hands out, scripted per case. */
const client = {
  getAccessToken: vi.fn<() => Promise<{ token?: string | null }>>(),
}
const auth = {
  getClient: vi.fn<() => Promise<typeof client>>(),
}
const constructed: unknown[] = []

vi.mock('google-auth-library', () => ({
  GoogleAuth: class {
    constructor(options: unknown) {
      constructed.push(options)
    }

    getClient = auth.getClient
  },
}))

const { adcCredentialSource } = await import('../src/auth.ts')

afterEach(() => {
  vi.clearAllMocks()
  constructed.length = 0
})

/** The LlmError a rejected token request carries. */
async function failure(promise: Promise<unknown>): Promise<LlmError> {
  try {
    await promise
  } catch (error: unknown) {
    if (error instanceof LlmError) return error
    throw error
  }
  throw new Error('expected the token request to fail')
}

describe('adcCredentialSource', () => {
  it('requests the configured scopes once and mints bearers through the discovered client', async () => {
    auth.getClient.mockResolvedValue(client)
    client.getAccessToken.mockResolvedValueOnce({ token: 'ya29.first' }).mockResolvedValueOnce({ token: 'ya29.second' })
    const source = adcCredentialSource(['scope-a', 'scope-b'])
    // The library loads, and discovery starts, on the first bearer request.
    expect(constructed).toEqual([])
    await expect(source.accessToken()).resolves.toBe('ya29.first')
    expect(constructed).toEqual([{ scopes: ['scope-a', 'scope-b'] }])
    await expect(source.accessToken()).resolves.toBe('ya29.second')
    expect(auth.getClient).toHaveBeenCalledTimes(1)
  })

  it('reports missing credentials with the login remedy and retries discovery on the next call', async () => {
    auth.getClient
      .mockRejectedValueOnce(new Error('Could not load the default credentials'))
      .mockResolvedValueOnce(client)
    client.getAccessToken.mockResolvedValue({ token: 'ya29.later' })
    const source = adcCredentialSource(['scope'])
    const missing = await failure(source.accessToken())
    expect(missing.code).toBe('MISSING_CREDENTIAL')
    expect(missing.message).toContain('gcloud auth application-default login')
    expect(missing.message).toContain('Could not load the default credentials')
    await expect(source.accessToken()).resolves.toBe('ya29.later')
    expect(auth.getClient).toHaveBeenCalledTimes(2)
  })

  it('maps a token refusal from discovered credentials to AUTH', async () => {
    auth.getClient.mockResolvedValue(client)
    client.getAccessToken.mockRejectedValueOnce(new Error('invalid_grant: Token has been expired or revoked.'))
    const source = adcCredentialSource(['scope'])
    const refused = await failure(source.accessToken())
    expect(refused.code).toBe('AUTH')
    expect(refused.message).toContain('invalid_grant')
  })

  it('refuses an empty or absent token as AUTH', async () => {
    auth.getClient.mockResolvedValue(client)
    client.getAccessToken.mockResolvedValueOnce({ token: '' }).mockResolvedValueOnce({ token: null })
    const source = adcCredentialSource(['scope'])
    await expect(source.accessToken()).rejects.toMatchObject({ code: 'AUTH' })
    await expect(source.accessToken()).rejects.toMatchObject({ code: 'AUTH' })
  })

  it('describes a non-Error rejection without losing its text', async () => {
    auth.getClient.mockRejectedValueOnce('plain failure')
    const source = adcCredentialSource(['scope'])
    expect((await failure(source.accessToken())).message).toContain('plain failure')
  })
})
