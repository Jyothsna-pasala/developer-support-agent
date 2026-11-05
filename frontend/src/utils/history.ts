import type { HistoryMessage } from '../api/types'
import type { ToolSummary, UiMessage } from '../conversation/types'

let counter = 0

/** Unique id for list keys (stable within the page's lifetime). */
export function nextId(prefix = 'm'): string {
  counter += 1
  return `${prefix}-${Date.now().toString(36)}-${counter}`
}

/** Convert GET /history entries into displayable messages, oldest first. */
export function historyToUiMessages(messages: HistoryMessage[]): UiMessage[] {
  return messages.map((message) => ({
    id: nextId('h'),
    role: message.role,
    content: message.content,
    timestamp: message.timestamp,
    origin: 'history',
  }))
}

// DeveloperSupportAgent.ask() records tool calls as
// f"{call.name}({json.dumps(call.arguments)}) -> {status}" with status ok|error.
const TOOL_SUMMARY = /^([A-Za-z0-9_.-]+)\((.*)\) -> (ok|error)$/s

/** Parse a history tool entry. Returns null if the format is not recognised. */
export function parseToolSummary(content: string): ToolSummary | null {
  const match = TOOL_SUMMARY.exec(content)
  if (!match) return null
  return { name: match[1], argumentsText: match[2], ok: match[3] === 'ok' }
}
