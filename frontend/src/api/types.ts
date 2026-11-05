// TypeScript mirrors of the FastAPI schemas in app/api/schemas.py.
// Field names and shapes match the backend exactly; nothing here is invented.

/** GET /health -> HealthOut */
export interface HealthResponse {
  status: 'ok'
  agent: string
  /** "ollama" or "local" in practice; the schema types it as a string. */
  mode: string
  /** Configured Ollama model tag in ollama mode, otherwise null. Not a liveness check. */
  model: string | null
  active_sessions: number
  max_sessions: number
  max_concurrent_turns: number
}

/** GET /tools -> ToolOut[] */
export interface ToolInfo {
  name: string
  description: string
  /** JSON Schema of the tool's input. */
  parameters: Record<string, unknown>
  output_description: string
}

/** POST /sessions -> SessionOut (201) */
export interface SessionResponse {
  session_id: string
  /** Display name of the backend serving this session. */
  backend: string
  /** ISO 8601 timestamp with offset. */
  created_at: string
}

/** POST /sessions/{id}/messages request body -> ChatRequest */
export interface ChatRequest {
  message: string
}

/** Tool executor result (app/tools.py execute_tool). */
export interface ToolSuccessOutput {
  tool: string
  ok: true
  result: unknown
}

export interface ToolFailureOutput {
  tool: string
  ok: false
  error: string
}

export type ToolOutput = ToolSuccessOutput | ToolFailureOutput

/** ToolCallOut */
export interface ToolCall {
  name: string
  arguments: Record<string, unknown>
  ok: boolean
  output: ToolOutput
}

/** POST /sessions/{id}/messages -> ChatResponse */
export interface ChatResponse {
  session_id: string
  reply: string
  tool_calls: ToolCall[]
  backend: string
}

export type HistoryRole = 'user' | 'assistant' | 'tool'

/** HistoryMessage */
export interface HistoryMessage {
  role: HistoryRole
  /** For "tool" entries this is a summary such as `name({"a": 1}) -> ok`. */
  content: string
  timestamp: string
}

/** GET /sessions/{id}/history -> HistoryOut */
export interface HistoryResponse {
  session_id: string
  messages: HistoryMessage[]
}

/** Error body for 404/409/503 and agent-side 422 (ErrorOut). */
export interface ErrorBody {
  detail: string
}

/** FastAPI request-validation error item (422). */
export interface ValidationErrorItem {
  loc: (string | number)[]
  msg: string
  type: string
  input?: unknown
  ctx?: Record<string, unknown>
}

export interface ValidationErrorBody {
  detail: ValidationErrorItem[]
}

// Known tool result shapes from app/tools.py. Used only for rendering, after
// checking the tool name and the shape at runtime.

export interface DateTimeResult {
  iso_8601: string
  date: string
  time: string
  weekday: string
  timezone: string
  utc_offset: string
}

export interface CalculationResult {
  expression: string
  result: number
}

export interface KnowledgeBaseArticle {
  category: string
  title: string
  content: string
  score: number
}

export interface KnowledgeBaseResult {
  query: string
  match_count: number
  results: KnowledgeBaseArticle[]
}
