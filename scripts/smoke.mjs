// End-to-end smoke: pack this plugin, install it into a fresh profile with the
// PUBLISHED dsh CLI, and run one headless prompt against a local mock of the
// Vertex OpenAI-compatible endpoint. Needs network for npm, no Google login.
import { execFileSync, spawn } from 'node:child_process'
import { createServer } from 'node:http'
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const DSH = process.env.DSH_SMOKE_CLI ?? '@deepseek-ai/dsh@0.1.5-rc.2'
const seen = []

const server = createServer((req, res) => {
  let body = ''
  req.on('data', (chunk) => { body += chunk })
  req.on('end', () => {
    seen.push({ url: req.url, authorization: req.headers.authorization, model: JSON.parse(body).model })
    res.writeHead(200, { 'content-type': 'text/event-stream' })
    const send = data => res.write(`data: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`)
    send({ choices: [{ index: 0, delta: { role: 'assistant', content: 'VERTEX_SMOKE_OK' } }] })
    send({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 } })
    send('[DONE]')
    res.end()
  })
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
const baseURL = `http://127.0.0.1:${server.address().port}/v1`

const work = mkdtempSync(join(tmpdir(), 'dsh-vertex-smoke-'))
const home = join(work, 'home')
const cwd = join(work, 'workspace')
execFileSync('mkdir', ['-p', home, cwd])
const env = { ...process.env, DSH_HOME: home, DSH_TELEMETRY_DISABLED: '1', GOOGLE_OAUTH_ACCESS_TOKEN: 'smoke-token', GOOGLE_CLOUD_PROJECT: '' }
// Asynchronous on purpose: the mock endpoint lives in this process, and a
// synchronous spawn would block the event loop it answers from.
const dsh = args => new Promise((resolve) => {
  const child = spawn('npx', ['-y', DSH, ...args], { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] })
  let stdout = ''
  let stderr = ''
  child.stdout.on('data', (chunk) => { stdout += chunk })
  child.stderr.on('data', (chunk) => { stderr += chunk })
  const timer = setTimeout(() => child.kill('SIGKILL'), 180_000)
  child.on('close', (status, signal) => {
    clearTimeout(timer)
    resolve({ status, signal, stdout, stderr })
  })
})

try {
  execFileSync('npm', ['pack', '--pack-destination', work], { cwd: root, stdio: 'ignore' })
  const tarball = join(work, readdirSync(work).find(name => name.endsWith('.tgz')))

  writeFileSync(join(home, 'settings.yaml'), `llm-vertex:\n  auth: access-token\n  baseURL: ${baseURL}\n`)
  writeFileSync(join(work, 'model.patch.yml'), [
    '- id: agent-default-model',
    "  name: '@deepseek-ai/dsh-agent-default-model'",
    '  config:',
    '    provider: vertex-ai',
    '    model: gemini-3.8-flash',
    '',
  ].join('\n'))

  const steps = [
    ['create profile', ['--profile', 'vertex-smoke', '--from-default-profile', 'headless', '--dump-config']],
    ['install plugin', ['plugin', '--profile', 'vertex-smoke', 'add', tarball]],
    ['run prompt', ['--profile', 'vertex-smoke', '--patch', join(work, 'model.patch.yml'), 'Reply with one word.']],
  ]
  let last
  for (const [label, args] of steps) {
    last = await dsh(args)
    if (last.status !== 0) {
      console.error(`smoke: "${label}" exited ${last.status ?? last.signal}\n${last.stdout}\n${last.stderr}`)
      console.error('smoke: requests seen by the mock endpoint:', seen)
      process.exitCode = 1
      break
    }
    console.log(`smoke: ${label} ok`)
  }
  if (process.exitCode !== 1) {
    const request = seen[0]
    const ok = last.stdout.includes('VERTEX_SMOKE_OK')
      && request?.url === '/v1/chat/completions'
      && request.authorization === 'Bearer smoke-token'
      && request.model === 'google/gemini-3.8-flash'
    console.log('smoke: request seen by the mock endpoint:', request)
    console.log(ok ? 'smoke: PASS' : `smoke: FAIL\nstdout:\n${last.stdout}\nstderr:\n${last.stderr}`)
    if (!ok) process.exitCode = 1
  }
} finally {
  server.close()
  rmSync(work, { recursive: true, force: true })
}
