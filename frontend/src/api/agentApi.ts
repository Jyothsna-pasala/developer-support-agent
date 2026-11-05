// One function per FastAPI endpoint (app/api/main.py). Every failure is
// rethrown as an ApiError, so callers never handle raw axios errors.

import { apiClient, MESSAGE_TIMEOUT_MS } from './client'
import { toApiError } from './errors'
import type {
  ChatRequest,
  ChatResponse,
  HealthResponse,
  HistoryResponse,
  SessionResponse,
  ToolInfo,
} from './types'

async function call<T>(request: () => Promise<{ data: T }>): Promise<T> {
  try {
    return (await request()).data
  } catch (error) {
    throw toApiError(error)
  }
}

const sessionPath = (sessionId: string) => `/sessions/${encodeURIComponent(sessionId)}`

export const agentApi = {
  /** GET /health */
  getHealth: (signal?: AbortSignal) => call(() => apiClient.get<HealthResponse>('/health', { signal })),

  /** GET /tools */
  getTools: (signal?: AbortSignal) => call(() => apiClient.get<ToolInfo[]>('/tools', { signal })),

  /** POST /sessions (201) */
  createSession: () => call(() => apiClient.post<SessionResponse>('/sessions')),

  /** POST /sessions/{id}/messages. Can take minutes with a local model. */
  sendMessage: (sessionId: string, message: string, signal?: AbortSignal) =>
    call(() =>
      apiClient.post<ChatResponse>(
        `${sessionPath(sessionId)}/messages`,
        { message } satisfies ChatRequest,
        { signal, timeout: MESSAGE_TIMEOUT_MS },
      ),
    ),

  /** GET /sessions/{id}/history */
  getHistory: (sessionId: string, signal?: AbortSignal) =>
    call(() => apiClient.get<HistoryResponse>(`${sessionPath(sessionId)}/history`, { signal })),

  /** POST /sessions/{id}/reset (204) */
  resetSession: async (sessionId: string): Promise<void> => {
    await call(() => apiClient.post<void>(`${sessionPath(sessionId)}/reset`))
  },

  /** DELETE /sessions/{id} (204) */
  deleteSession: async (sessionId: string): Promise<void> => {
    await call(() => apiClient.delete<void>(sessionPath(sessionId)))
  },
}

export type AgentApi = typeof agentApi
