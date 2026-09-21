# dsh-vertex-gemini

Use Gemini on **Google Cloud Vertex AI** from [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`), signed in with your own Google account. No API key is created or stored: the plugin uses the Google login already on your machine (Application Default Credentials) and bills the Google Cloud project you name.

It works with the official `npx @deepseek-ai/dsh` — no fork and no build.

## What you need

- Node.js 22.19+ or 24+.
- The [gcloud CLI](https://cloud.google.com/sdk/docs/install).
- [pnpm](https://pnpm.io/installation) on your PATH (`npm install -g pnpm`): the `dsh plugin` command runs it inside the profile.
- Read access to this repository and the [GitHub CLI](https://cli.github.com) signed in (`gh auth login`), to download releases.
- A Google Cloud project with the Vertex AI API enabled, and permission to call it (`roles/aiplatform.user`).

## Set up (about two minutes)

1. Sign in once. This stores a Google login that the plugin refreshes on its own:

   ```bash
   gcloud auth application-default login
   ```

2. Download the latest release into a folder you will keep, then install it into the Web profile. The profile records the tarball's absolute path, so a file deleted from Downloads breaks the next profile install; `~/.dsh/plugins` is a safe home for it:

   ```bash
   mkdir -p ~/.dsh/plugins
   gh release download --repo mn288/dsh-vertex-gemini --pattern '*.tgz' --dir ~/.dsh/plugins
   npx @deepseek-ai/dsh plugin --profile web add ~/.dsh/plugins/dsh-vertex-gemini-0.1.0.tgz
   ```

   pnpm ends with a block of `missing peer @deepseek-ai/...` warnings. They are expected: those packages live one directory up in the profile tree, where Node finds them at run time. The line to look for is `+ dsh-vertex-gemini` under `dependencies`.

   To use the plugin from the terminal as well, repeat the `add` command with `--profile headless`. To update later, download the newer release into the same folder and run the same `add` command with the new file.

3. Name your Google Cloud project. Either export it in the shell that starts `dsh`:

   ```bash
   export GOOGLE_CLOUD_PROJECT=my-gcp-project
   ```

   or save it in `$DSH_HOME/settings.yaml` (by default `~/.dsh/settings.yaml`):

   ```yaml
   llm-vertex:
     project: my-gcp-project
   ```

4. Start the Web UI and pick the model:

   ```bash
   npx @deepseek-ai/dsh web
   ```

   If `dsh web` was already running, stop it and start it again: plugins are loaded at startup. Then open the model picker and choose **Google Vertex AI → Gemini 3.8 Flash**. The choice becomes the default for new sessions.

   The picker offers the reasoning levels Low, Medium and High. A default of `xhigh` or `max` left over from another provider in `settings.yaml` makes every request fail with `UNSUPPORTED_REASONING_EFFORT`; pick a level in the picker or set `reasoningEffort: high` under `agent-default-model`.

Until a project is set, the provider is listed under **Settings → Models** but no model appears in the picker. That is expected: the route stays off until it knows which project to call.

To remove the plugin: `npx @deepseek-ai/dsh plugin --profile web remove dsh-vertex-gemini`.

## Settings

Every field is optional and goes under `llm-vertex:` in `settings.yaml`. Changes apply to the next request; no restart is needed.

| Field | Default | Meaning |
|---|---|---|
| `project` | `$GOOGLE_CLOUD_PROJECT`, then `$GCLOUD_PROJECT` | Google Cloud project that is called and billed |
| `location` | `global` | Vertex AI location; a region such as `europe-west1` uses that region's endpoint |
| `reasoningEffort` | `high` | Default thinking level: `low`, `medium`, or `high` |
| `models` | Gemini 3.8 Flash | Models shown in the picker; each entry takes `id`, and optionally `name`, `description`, `contextWindow`, `maxTokens` |
| `maxTokens` | `65536` | Output cap for a model that names none |
| `defaultContextWindow` | `1048576` | Context size for a model that names none |
| `publisher` | `google` | Prefix added to a bare model id on the wire (`gemini-3.8-flash` is sent as `google/gemini-3.8-flash`) |
| `baseURL` | computed from project and location | Full OpenAI-compatible endpoint root, for a company gateway |
| `auth` | `adc` | `adc` uses the Google login; `access-token` reads a bearer you supply |
| `accessTokenEnv` | `GOOGLE_OAUTH_ACCESS_TOKEN` | Where the bearer comes from with `auth: access-token` (for CI jobs that mint their own token) |
| `streamIdleTimeoutMs` | `300000` | How long one silent stretch of a response may last before the request fails |
| `retryPolicy` | five retries | Retry behaviour for failed model requests |

Adding a second model:

```yaml
llm-vertex:
  project: my-gcp-project
  models:
    - id: gemini-3.8-flash
      name: Gemini 3.8 Flash
    - id: another-gemini-model-id   # any model id your project can call
      name: Shown in the picker
```

Any Gemini model id your project can call works even when it is not listed; the list only feeds the picker.

## When something fails

| Message | Cause and fix |
|---|---|
| `MISSING_CREDENTIAL` … run "gcloud auth application-default login" | No Google login on this machine, or it expired. Run the command and retry; no restart needed. |
| `AUTH` | The login exists but Google refused it (revoked, or the account lacks access). Sign in again, and check the account has `roles/aiplatform.user` on the project. |
| `NOT_FOUND` | The project cannot use that model, or the model id or location is wrong. Check the id, try `location: global`, and confirm the Vertex AI API is enabled. |
| `QUOTA` / `RATE_LIMIT` | The project's Gemini quota is exhausted or throttled. The harness retries on its own; raise the quota in the Cloud console if it persists. |
| No Vertex model in the picker | No project is configured. See step 3. |
| `UNSUPPORTED_CONTENT` | You attached an image. This plugin sends text only. |

## Verification status

Verified on 2026-09-21 against the real Vertex AI API (project `common-ai-tooling-prd`, location `global`, Gemini 3.8 Flash): the real-API test below (a text answer, then one tool-call round trip with the signature replayed) passes, and a headless `dsh` session installed from the release tarball read a file through a tool and answered correctly. Unit tests, a Loader boot test, and an install-and-run smoke against a local mock also pass.

Re-run the real-API test with a valid Google login whenever the adapter or the harness version changes:

```bash
DSH_VERTEX_E2E=1 DSH_VERTEX_PROJECT=my-gcp-project pnpm exec vitest run tests/adapter.e2e.ts
```

Gemini 3.8 Flash counts its thinking tokens against the output cap: a request with a very small `maxTokens` can finish with `max-tokens` and no text. Leave the cap at the default (65536) or at least in the low thousands.

## Limitations

- **Text only.** Image input is refused rather than silently dropped.
- **No settings form.** The Models page lists the provider but has no dedicated card for it; configure it through the environment variable or `settings.yaml` as shown above.
- **Tested against `@deepseek-ai/dsh` 0.1.5-rc.2.** The harness APIs are pre-stable, so a later release may need a plugin update.

## How it works

The plugin registers one provider route, `vertex-ai`, and talks to Vertex AI's OpenAI-compatible chat-completions endpoint with streaming. Two details matter for agent work:

- **Thought signatures.** Gemini 3 attaches an opaque signature to a tool call and rejects the follow-up request if it is not sent back. The plugin stores each signature next to the tool call in the session log and echoes it on later requests, so multi-step tool use survives restarts and resumed sessions. History produced by another model or provider is sent with Google's documented skip marker instead.
- **Credentials.** `google-auth-library` finds the login in Google's standard order (`GOOGLE_APPLICATION_CREDENTIALS`, the gcloud user login, then the metadata server), caches the hourly token, and refreshes it. The library is loaded on the first request only, so the plugin costs nothing at startup when you are using another provider.

## Development

```bash
pnpm install
pnpm test          # unit tests, no network or Google login needed
pnpm run typecheck
pnpm run smoke     # packs the plugin, installs it with the published dsh CLI, and runs one prompt against a local mock endpoint
```

`tests/adapter.e2e.ts` calls the real API. Run it with a valid Google login:

```bash
DSH_VERTEX_E2E=1 DSH_VERTEX_PROJECT=my-gcp-project pnpm exec vitest run tests/adapter.e2e.ts
```

### Sharing it

The repository is private and `package.json` carries `"private": true`, so an accidental `npm publish` is refused. Colleagues need read access first: add them under the repository's **Settings → Collaborators**. To hand the plugin to them:

- **As a release (current setup):** bump `version`, run `npm pack`, and attach the `.tgz` to a new release with `gh release create v<version> dsh-vertex-gemini-<version>.tgz`. Colleagues with read access to the repository install it as shown in the setup steps.
- **Through an internal registry:** remove `"private": true`, point `publishConfig.registry` at the registry, and publish there; colleagues then install it by name.

## License

MIT. The adapter follows the structure of the DeepSeek Harness LLM adapters, which are MIT licensed by DeepSeek; see [LICENSE](LICENSE).
