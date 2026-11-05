import type { CalculationResult, DateTimeResult, KnowledgeBaseResult } from '../api/types'

// Runtime checks before rendering a tool result with a dedicated view. If the
// backend ever returns a different shape, the UI falls back to raw JSON.

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isCalculationResult(value: unknown): value is CalculationResult {
  return isRecord(value) && typeof value.expression === 'string' && typeof value.result === 'number'
}

export function isDateTimeResult(value: unknown): value is DateTimeResult {
  return (
    isRecord(value) &&
    ['iso_8601', 'date', 'time', 'weekday', 'timezone', 'utc_offset'].every((k) => typeof value[k] === 'string')
  )
}

export function isKnowledgeBaseResult(value: unknown): value is KnowledgeBaseResult {
  return (
    isRecord(value) &&
    typeof value.query === 'string' &&
    typeof value.match_count === 'number' &&
    Array.isArray(value.results) &&
    value.results.every(
      (r) => isRecord(r) && typeof r.title === 'string' && typeof r.category === 'string' && typeof r.content === 'string',
    )
  )
}

/** Pretty JSON for display. */
export function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2) ?? String(value)
  } catch {
    return String(value)
  }
}
