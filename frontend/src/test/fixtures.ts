// Response bodies copied from the real FastAPI backend (README section 11.4
// and the local-mode API tests), so tests use the actual shapes.

import { vi } from 'vitest'

import type { AgentApi } from '../api/agentApi'
import type { ChatResponse, HealthResponse, HistoryResponse, SessionResponse, ToolInfo } from '../api/types'

export const SESSION: SessionResponse = {
  session_id: 'a027ba84d7cc475f9026edd47bac4256',
  backend: 'Ollama local LLM (qwen3:4b); rule-based fallback enabled',
  created_at: '2026-11-04T12:17:58.661957+05:30',
}

export const CALC_REPLY: ChatResponse = {
  session_id: SESSION.session_id,
  reply: 'The result of `(1200 * 0.18) + 50` is **266**.',
  tool_calls: [
    {
      name: 'calculate_expression',
      arguments: { expression: '(1200 * 0.18) + 50' },
      ok: true,
      output: {
        tool: 'calculate_expression',
        ok: true,
        result: { expression: '(1200 * 0.18) + 50', result: 266 },
      },
    },
  ],
  backend: SESSION.backend,
}

export const FALLBACK_REPLY: ChatResponse = {
  session_id: SESSION.session_id,
  reply:
    '[LLM unavailable: Ollama returned HTTP 500: simulated failure Answered by the rule-based fallback.]\n' +
    '6 * 7 = 42',
  tool_calls: [
    {
      name: 'calculate_expression',
      arguments: { expression: '6 * 7' },
      ok: true,
      output: { tool: 'calculate_expression', ok: true, result: { expression: '6 * 7', result: 42 } },
    },
  ],
  backend: SESSION.backend,
}

export const HISTORY: HistoryResponse = {
  session_id: SESSION.session_id,
  messages: [
    { role: 'user', content: 'Calculate (1200 * 0.18) + 50', timestamp: '2026-11-04T12:18:00.000000+05:30' },
    {
      role: 'tool',
      content: 'calculate_expression({"expression": "(1200 * 0.18) + 50"}) -> ok',
      timestamp: '2026-11-04T12:20:14.000000+05:30',
    },
    {
      role: 'assistant',
      content: 'The result of `(1200 * 0.18) + 50` is **266**.',
      timestamp: '2026-11-04T12:20:14.100000+05:30',
    },
  ],
}

export const HEALTH: HealthResponse = {
  status: 'ok',
  agent: 'Developer Support Agent',
  mode: 'ollama',
  model: 'qwen3:4b',
  active_sessions: 0,
  max_sessions: 20,
  max_concurrent_turns: 1,
}

export const TOOLS: ToolInfo[] = [
  {
    name: 'get_current_datetime',
    description:
      'Get the current local date, time, weekday and time zone. Use this whenever the user asks about the current date or time.',
    parameters: { type: 'object', properties: {}, required: [], additionalProperties: false },
    output_description: 'Object with iso_8601, date, time, weekday, timezone and utc_offset strings.',
  },
  {
    name: 'calculate_expression',
    description: 'Safely evaluate a basic arithmetic expression.',
    parameters: {
      type: 'object',
      properties: { expression: { type: 'string', description: "The arithmetic expression, e.g. '(15 * 4) / 3'." } },
      required: ['expression'],
      additionalProperties: false,
    },
    output_description: 'Object with the original expression and numeric result.',
  },
]

/** An AgentApi whose every method is a vi.fn(); override per test. */
export function createFakeApi(overrides: Partial<AgentApi> = {}): { [K in keyof AgentApi]: ReturnType<typeof vi.fn> } & AgentApi {
  return {
    getHealth: vi.fn().mockResolvedValue(HEALTH),
    getTools: vi.fn().mockResolvedValue(TOOLS),
    createSession: vi.fn().mockResolvedValue(SESSION),
    sendMessage: vi.fn().mockResolvedValue(CALC_REPLY),
    getHistory: vi.fn().mockResolvedValue(HISTORY),
    resetSession: vi.fn().mockResolvedValue(undefined),
    deleteSession: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  } as never
}

/** A promise you resolve or reject from the test. */
export function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}
