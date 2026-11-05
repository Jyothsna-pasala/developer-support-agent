import { useCallback, useEffect, useState } from 'react'

import { agentApi } from '../api/agentApi'
import { toApiError, type ApiError } from '../api/errors'
import type { HealthResponse } from '../api/types'

export const HEALTH_POLL_MS = 30_000

export interface HealthState {
  health: HealthResponse | null
  error: ApiError | null
  loading: boolean
  refresh: () => void
}

/**
 * Poll GET /health. Note that /health does not contact Ollama: it proves the
 * API is up and reports the configured mode and model only.
 */
export function useHealth(pollMs: number = HEALTH_POLL_MS): HealthState {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    agentApi
      .getHealth(controller.signal)
      .then((body) => {
        setHealth(body)
        setError(null)
      })
      .catch((caught: unknown) => {
        const apiError = toApiError(caught)
        if (apiError.kind !== 'aborted') setError(apiError)
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    const timer = window.setTimeout(() => setTick((t) => t + 1), pollMs)
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [pollMs, tick])

  const refresh = useCallback(() => setTick((t) => t + 1), [])
  return { health, error, loading, refresh }
}
