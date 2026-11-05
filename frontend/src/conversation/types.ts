import type { ToolCall } from '../api/types'

/**
 * A message as displayed in the chat.
 *
 * - `live` messages come from POST /messages and carry full tool calls.
 * - `history` messages come from GET /history after a refresh; tool entries
 *   there are summary strings only (the API returns no structured output).
 */
export interface UiMessage {
  id: string
  role: 'user' | 'assistant' | 'tool' | 'divider'
  content: string
  /** ISO 8601. Server time for history, browser time for live messages. */
  timestamp: string
  origin: 'live' | 'history'
  /** Live assistant replies only. */
  toolCalls?: ToolCall[]
  /** User messages: "failed" = not processed, "unconfirmed" = sent but no reply received. */
  status?: 'failed' | 'unconfirmed'
}

/** A tool entry from GET /history, parsed from `name({...}) -> ok|error`. */
export interface ToolSummary {
  name: string
  argumentsText: string
  ok: boolean
}
