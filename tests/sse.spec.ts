import { describe, expect, it, vi } from 'vitest'
import { DONE, parseSse } from '../src/sse.ts'

function bytes(text: string): ReadableStream<BufferSource> {
  return new ReadableStream<BufferSource>({
    start(controller) {
      // `slice` yields a Uint8Array over a plain ArrayBuffer, the BufferSource the stream is typed for.
      controller.enqueue(new TextEncoder().encode(text).slice())
      controller.close()
    },
  })
}

async function collect(stream: ReadableStream<BufferSource>, onComment?: (comment: string) => void): Promise<string[]> {
  const payloads: string[] = []
  for await (const payload of parseSse(stream, onComment)) payloads.push(payload)
  return payloads
}

describe('parseSse', () => {
  it('yields each data payload, reports comments as activity, and stops at [DONE]', async () => {
    const onComment = vi.fn()
    const payloads = await collect(bytes(': keep-alive\n\ndata: {"a":1}\n\ndata: [DONE]\n\ndata: {"after":true}\n\n'), onComment)
    expect(payloads).toEqual(['{"a":1}', DONE])
    expect(onComment).toHaveBeenCalledWith('keep-alive')
  })

  it('parses without an activity callback', async () => {
    await expect(collect(bytes('data: x\n\ndata: [DONE]\n\n'))).resolves.toEqual(['x', DONE])
  })

  it('fails with STREAM_CLOSED when the bytes end before [DONE]', async () => {
    await expect(collect(bytes('data: x\n\n'))).rejects.toMatchObject({ code: 'STREAM_CLOSED' })
  })
})
