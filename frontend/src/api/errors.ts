import axios from 'axios'

import { API_BASE_URL } from './client'
import type { ValidationErrorItem } from './types'

export type ApiErrorKind =
  | 'network' // No response: backend down, wrong URL or a CORS rejection.
  | 'timeout'
  | 'aborted'
  | 'not_found' // 404: unknown, deleted or expired session.
  | 'busy' // 409: the session is processing another request.
  | 'validation' // 422
  | 'unavailable' // 503: model busy, session limit, or Ollama unavailable.
  | 'server' // Anything else, including plain-text 500s.

/** The only error type components see. `message` is safe to display. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status?: number

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
  }

  /** True when there was no HTTP response at all. */
  get isOffline(): boolean {
    return this.kind === 'network'
  }
}

function isValidationItem(value: unknown): value is ValidationErrorItem {
  return typeof value === 'object' && value !== null && typeof (value as { msg?: unknown }).msg === 'string'
}

/**
 * Read FastAPI's `detail`, which is a string for most errors and an array of
 * validation items for request-validation 422s. Returns null for other bodies
 * (for example the plain-text "Internal Server Error").
 */
export function readDetail(data: unknown): string | null {
  if (typeof data !== 'object' || data === null || !('detail' in data)) return null
  const detail = (data as { detail: unknown }).detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail.filter(isValidationItem).map((item) => item.msg.replace(/^Value error, /, ''))
    return messages.length ? messages.join(' ') : null
  }
  return null
}

const KIND_BY_STATUS: Record<number, ApiErrorKind> = {
  404: 'not_found',
  409: 'busy',
  422: 'validation',
  503: 'unavailable',
}

/** Convert anything thrown by an API call into an {@link ApiError}. */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  if (axios.isCancel(error)) return new ApiError('aborted', 'The request was cancelled.')
  if (!axios.isAxiosError(error)) {
    return new ApiError('server', 'An unexpected error occurred.')
  }
  if (error.code === 'ERR_CANCELED') return new ApiError('aborted', 'The request was cancelled.')

  const response = error.response
  if (!response) {
    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      return new ApiError('timeout', 'The request took too long and was stopped by the browser.')
    }
    return new ApiError(
      'network',
      `Cannot reach the API at ${API_BASE_URL}. Is "python -m app.api" running? ` +
        '(A CORS rejection looks the same: open the app on http://localhost:5173.)',
    )
  }

  const status = response.status
  const detail = readDetail(response.data)
  const kind = KIND_BY_STATUS[status] ?? 'server'
  return new ApiError(kind, detail ?? `The API returned an unexpected error (HTTP ${status}).`, status)
}
