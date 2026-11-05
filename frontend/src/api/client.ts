import axios from 'axios'

const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'

/** Base URL of the FastAPI backend, from VITE_API_BASE_URL. */
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL?.trim() || DEFAULT_BASE_URL).replace(
  /\/+$/,
  '',
)

/** Timeout for quick calls (health, tools, sessions, history). */
export const DEFAULT_TIMEOUT_MS = 15_000

/**
 * Timeout for one agent turn: up to API_QUEUE_TIMEOUT_SECONDS (300 s) waiting
 * for a model slot, plus a qwen3:4b turn that can take several minutes on a CPU.
 */
export const MESSAGE_TIMEOUT_MS = 15 * 60_000

// No custom headers: the API's CORS policy only allows Content-Type, which
// axios sets itself for JSON bodies.
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: DEFAULT_TIMEOUT_MS,
})
