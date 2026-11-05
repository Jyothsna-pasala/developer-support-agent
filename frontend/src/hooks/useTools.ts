import { useCallback, useEffect, useState } from 'react'

import { agentApi } from '../api/agentApi'
import { toApiError, type ApiError } from '../api/errors'
import type { ToolInfo } from '../api/types'

export interface ToolsState {
  tools: ToolInfo[] | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

/** Load GET /tools. */
export function useTools(): ToolsState {
  const [tools, setTools] = useState<ToolInfo[] | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    agentApi
      .getTools(controller.signal)
      .then((body) => {
        setTools(body)
        setError(null)
      })
      .catch((caught: unknown) => {
        const apiError = toApiError(caught)
        if (apiError.kind !== 'aborted') setError(apiError)
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [attempt])

  const reload = useCallback(() => {
    setLoading(true)
    setAttempt((a) => a + 1)
  }, [])

  return { tools, error, loading, reload }
}
