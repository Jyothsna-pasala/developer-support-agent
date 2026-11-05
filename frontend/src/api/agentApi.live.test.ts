// @vitest-environment-options {"url": "http://localhost:5174/"}
/**
 * Opt-in tests against a REAL running backend (skipped by default).
 *
 * They use the real axios client from a jsdom page whose origin is
 * http://localhost:5174, so jsdom enforces the API's CORS policy just like a
 * browser. Start the API with that origin allowed, then run (PowerShell):
 *
 *   $env:API_CORS_ORIGINS = "http://localhost:5174"
 *   python -m app.api --mode local --port 8001          # terminal 1
 *   $env:LIVE_API_URL = "http://127.0.0.1:8001"; npx vitest run src/api/agentApi.live.test.ts
 *
 * LIVE_OLLAMA_API_URL points at an API in ollama mode; that test needs a real
 * model and takes minutes on a CPU.
 */
import { describe, expect, it } from 'vitest'

import { agentApi } from './agentApi'
import { apiClient } from './client'
import { ApiError } from './errors'

// Vitest runs in Node; the app tsconfig deliberately has no Node types.
declare const process: { env: Record<string, string | undefined> }

const LOCAL_URL = process.env.LIVE_API_URL
const OLLAMA_URL = process.env.LIVE_OLLAMA_API_URL

async function expectApiError(promise: Promise<unknown>, kind: ApiError['kind'], status: number) {
  const error = await promise.then(
    () => null,
    (caught: unknown) => caught,
  )
  expect(error).toBeInstanceOf(ApiError)
  expect((error as ApiError).kind).toBe(kind)
  expect((error as ApiError).status).toBe(status)
  return error as ApiError
}

describe.skipIf(!LOCAL_URL)('live API (local mode)', () => {
  it('runs a full conversation through CORS with the real backend', async () => {
    expect(window.location.origin).toBe('http://localhost:5174')
    apiClient.defaults.baseURL = LOCAL_URL

    const health = await agentApi.getHealth()
    expect(Object.keys(health).sort()).toEqual(
      ['active_sessions', 'agent', 'max_concurrent_turns', 'max_sessions', 'mode', 'model', 'status'].sort(),
    )
    expect(health.mode).toBe('local')

    const tools = await agentApi.getTools()
    expect(tools.map((t) => t.name)).toEqual(['get_current_datetime', 'calculate_expression', 'search_knowledge_base'])

    const session = await agentApi.createSession()
    expect(session.backend).toBe('Local simulation (offline, no LLM)')

    const reply = await agentApi.sendMessage(session.session_id, 'calculate (1200 * 0.18) + 50')
    expect(reply.reply).toBe('(1200 * 0.18) + 50 = 266')
    expect(Object.keys(reply.tool_calls[0]).sort()).toEqual(['arguments', 'name', 'ok', 'output'])
    expect(reply.tool_calls[0].output).toEqual({
      tool: 'calculate_expression',
      ok: true,
      result: { expression: '(1200 * 0.18) + 50', result: 266 },
    })

    const kb = await agentApi.sendMessage(session.session_id, 'How do I fix a CORS error?')
    expect(kb.tool_calls[0].name).toBe('search_knowledge_base')

    const history = await agentApi.getHistory(session.session_id)
    expect(history.messages.map((m) => m.role)).toEqual(['user', 'tool', 'assistant', 'user', 'tool', 'assistant'])
    expect(history.messages[1].content).toBe('calculate_expression({"expression": "(1200 * 0.18) + 50"}) -> ok')

    const invalid = await expectApiError(agentApi.sendMessage(session.session_id, '   '), 'validation', 422)
    expect(invalid.message).toBe('Please enter a question.')

    await agentApi.resetSession(session.session_id)
    expect((await agentApi.getHistory(session.session_id)).messages).toEqual([])

    await agentApi.deleteSession(session.session_id)
    const gone = await expectApiError(agentApi.getHistory(session.session_id), 'not_found', 404)
    expect(gone.message).toBe('Session not found. It may have expired or been deleted.')
  })
})

describe.skipIf(!OLLAMA_URL)('live API (ollama mode, real model)', () => {
  it(
    'the model calls the calculator and returns its result',
    async () => {
      apiClient.defaults.baseURL = OLLAMA_URL
      const session = await agentApi.createSession()
      expect(session.backend).toContain('Ollama local LLM')
      const reply = await agentApi.sendMessage(session.session_id, 'Calculate (1200 * 0.18) + 50')
      const call = reply.tool_calls.find((c) => c.name === 'calculate_expression')
      expect(call?.output).toMatchObject({ ok: true, result: { result: 266 } })
      expect(reply.reply).toContain('266')
      await agentApi.deleteSession(session.session_id)
    },
    10 * 60_000,
  )
})
