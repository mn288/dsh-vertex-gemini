import { createServer } from 'node:http'
import type { IncomingMessage, Server, ServerResponse } from 'node:http'

/** One scripted behavior for the next request the mock server receives. */
export type Behavior =
  | { kind: 'sse'; events: string[]; delayMs?: number }
  | { kind: 'http-error'; status: number; body: string; contentType?: string; headers?: Record<string, string> }
  | { kind: 'close-early'; events: string[] }
  | { kind: 'empty-body' }

export interface MockServer {
  url: string
  /** Bodies of received requests, in order. */
  requests: unknown[]
  /** Header bags of received requests, in order (parallel to `requests`). */
  headers: IncomingMessage['headers'][]
  /** Request paths, in order (parallel to `requests`). */
  paths: string[]
  script: Behavior[]
  close(): Promise<void>
}

const servers: Server[] = []

/** Close every server opened since the last call; run from each spec's afterEach. */
export async function closeMockServers(): Promise<void> {
  await Promise.all(servers.splice(0).map(server => new Promise(resolve => server.close(resolve))))
}

/** A minimal complete text generation, as Vertex streams it. */
export const textEvents = [
  '{"choices":[{"delta":{"content":"","role":"assistant"},"index":0}],"model":"google/gemini-3.8-flash"}',
  '{"choices":[{"delta":{"content":"hello"},"index":0}]}',
  '{"choices":[{"delta":{"content":""},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}',
  '[DONE]',
]

/** One tool call carrying a thought signature, then a usage-only trailer. */
export const SIGNATURE = 'AY89a1+signature+base64=='
export const toolCallEvents = [
  `{"choices":[{"delta":{"role":"assistant","tool_calls":[{"extra_content":{"google":{"thought_signature":"${SIGNATURE}"}},"function":{"arguments":"{\\"city\\":\\"Paris\\"}","name":"get_weather"},"id":"call_1","index":0,"type":"function"}]},"index":0}]}`,
  '{"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":"tool_calls","index":0}]}',
  '{"choices":[],"usage":{"prompt_tokens":33,"completion_tokens":16,"total_tokens":97,"completion_tokens_details":{"reasoning_tokens":48}}}',
  '[DONE]',
]

/** Local Vertex OpenAI-compatible stand-in: replays scripted behaviors per request. */
export async function mockServer(script: Behavior[]): Promise<MockServer> {
  const requests: unknown[] = []
  const headers: IncomingMessage['headers'][] = []
  const paths: string[] = []
  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    const chunks: Buffer[] = []
    request.on('data', (chunk: Buffer) => { chunks.push(chunk) })
    request.on('end', () => {
      requests.push(JSON.parse(Buffer.concat(chunks).toString('utf8')))
      headers.push(request.headers)
      paths.push(request.url ?? '/')
      const behavior = script.shift()
      if (!behavior) {
        response.writeHead(500).end('mock script exhausted')
        return
      }
      if (behavior.kind === 'http-error') {
        response.writeHead(behavior.status, {
          'content-type': behavior.contentType ?? 'application/json',
          ...behavior.headers,
        })
        response.end(behavior.body)
        return
      }
      if (behavior.kind === 'empty-body') {
        response.writeHead(200, { 'content-type': 'text/event-stream' })
        response.end()
        return
      }
      response.writeHead(200, { 'content-type': 'text/event-stream' })
      // Vertex opens its streams with a keep-alive comment; the adapter counts it as transport activity.
      response.write(': keep-alive\n\n')
      const write = (index: number): void => {
        if (index >= behavior.events.length) {
          if (behavior.kind === 'sse') response.end()
          else response.destroy() // close-early: drop the socket mid-stream
          return
        }
        response.write(`data: ${behavior.events[index]}\n\n`)
        setTimeout(() => { write(index + 1) }, behavior.kind === 'sse' ? behavior.delayMs ?? 0 : 5)
      }
      write(0)
    })
  })
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('no port')
  return {
    url: `http://127.0.0.1:${address.port}`,
    requests,
    headers,
    paths,
    script,
    close: () => new Promise(resolve => server.close(() => { resolve() })),
  }
}
