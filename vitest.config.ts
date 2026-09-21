import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: { include: ['tests/**/*.spec.ts', 'tests/**/*.e2e.ts'], testTimeout: 20_000 },
})
