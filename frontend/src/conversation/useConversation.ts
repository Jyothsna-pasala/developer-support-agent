import { useCallback, useEffect, useReducer, useRef } from 'react'

import { agentApi, type AgentApi } from '../api/agentApi'
import { toApiError } from '../api/errors'
import { historyToUiMessages, nextId } from '../utils/history'
import { clearSession, loadSession, saveSession } from '../utils/storage'
import { validateMessage } from '../utils/validation'
import { conversationReducer, initialState, type ConversationState } from './reducer'
import type { UiMessage } from './types'

/** How often GET /history is retried while an earlier turn is still running (409). */
export const BUSY_POLL_MS = 10_000

export interface Conversation {
  state: ConversationState
  /** True when a new message may be sent now. */
  canSend: boolean
  /**
   * Send a message. Returns false at once if it was ignored (a turn is
   * running or the text is invalid); the reply arrives later in `state`.
   */
  send: (text: string) => boolean
  /** Retry the last failed action (resend a message or reload history). */
  retry: () => void
  /** Stop waiting for the current reply. The server-side turn keeps running. */
  stopWaiting: () => void
  /** Reload the conversation from GET /history. */
  refresh: () => void
  /** POST /reset: clear the transcript and model context, keep the session. */
  reset: () => Promise<void>
  /** DELETE the session and start over. */
  newConversation: () => void
  dismissError: () => void
  dismissNotice: () => void
}

export function useConversation(api: AgentApi = agentApi): Conversation {
  const [state, dispatch] = useReducer(conversationReducer, null, () => initialState(loadSession()))
  // Synchronous guard against double submits (state updates are asynchronous).
  const inFlight = useRef(false)
  const sendController = useRef<AbortController | null>(null)

  const restore = useCallback(
    async (sessionId: string, options: { quiet?: boolean; signal?: AbortSignal } = {}) => {
      if (!options.quiet) dispatch({ type: 'restoreStarted' })
      try {
        const history = await api.getHistory(sessionId, options.signal)
        dispatch({ type: 'restored', messages: historyToUiMessages(history.messages) })
      } catch (caught) {
        const error = toApiError(caught)
        if (error.kind === 'aborted') return
        if (error.kind === 'not_found') {
          clearSession()
          dispatch({ type: 'sessionLost', during: 'restore' })
        } else if (error.kind === 'busy') {
          dispatch({ type: 'turnStillRunning' })
        } else {
          dispatch({ type: 'restoreFailed', error })
        }
      }
    },
    [api],
  )

  // Restore the stored session once on load. StrictMode runs this twice in
  // development; the first request is aborted and ignored.
  const initialSessionId = useRef(state.session?.session_id)
  useEffect(() => {
    const sessionId = initialSessionId.current
    if (!sessionId) return
    const controller = new AbortController()
    void restore(sessionId, { signal: controller.signal })
    return () => controller.abort()
  }, [restore])

  // While an earlier turn is still running, poll until history is readable.
  const sessionId = state.session?.session_id
  useEffect(() => {
    if (state.phase !== 'waitingForTurn' || !sessionId) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      void restore(sessionId, { quiet: true, signal: controller.signal })
    }, BUSY_POLL_MS)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [state.phase, sessionId, restore])

  const canSend = state.phase === 'idle'

  const runTurn = useCallback(
    async (userMessage: UiMessage, session: ConversationState['session']) => {
      const controller = new AbortController()
      sendController.current = controller
      try {
        if (!session) {
          // Sessions are created on the first message, not on page load, so
          // visits do not use up the API's session limit.
          session = await api.createSession()
          saveSession(session)
          dispatch({ type: 'sessionCreated', session })
        }
        const response = await api.sendMessage(session.session_id, userMessage.content, controller.signal)
        dispatch({
          type: 'replyReceived',
          userMessageId: userMessage.id,
          reply: {
            id: nextId('a'),
            role: 'assistant',
            content: response.reply,
            timestamp: new Date().toISOString(),
            origin: 'live',
            toolCalls: response.tool_calls,
          },
        })
      } catch (caught) {
        const error = toApiError(caught)
        const failure = { messageId: userMessage.id, text: userMessage.content }
        if (error.kind === 'not_found') {
          clearSession()
          dispatch({ type: 'sessionLost', during: 'send', ...failure })
        } else {
          dispatch({ type: 'sendFailed', error, ...failure })
        }
      } finally {
        inFlight.current = false
        sendController.current = null
      }
    },
    [api],
  )

  const sendText = useCallback(
    (text: string, replaceId?: string): boolean => {
      if (inFlight.current || state.phase !== 'idle' || validateMessage(text) !== null) return false
      inFlight.current = true
      const userMessage: UiMessage = {
        id: nextId('u'),
        role: 'user',
        content: text.trim(),
        timestamp: new Date().toISOString(),
        origin: 'live',
      }
      dispatch({ type: 'sendStarted', message: userMessage, replaceId, at: Date.now() })
      void runTurn(userMessage, state.session)
      return true
    },
    [runTurn, state.phase, state.session],
  )

  const send = useCallback((text: string) => sendText(text), [sendText])

  const refresh = useCallback(() => {
    if (sessionId && state.phase !== 'sending') void restore(sessionId)
  }, [restore, sessionId, state.phase])

  const retry = useCallback(() => {
    const action = state.retry
    if (!action) return
    if (action.kind === 'send') sendText(action.text, action.messageId)
    else refresh()
  }, [state.retry, sendText, refresh])

  const stopWaiting = useCallback(() => sendController.current?.abort(), [])

  const reset = useCallback(async () => {
    if (!state.session || state.phase !== 'idle') return
    dispatch({ type: 'resetStarted' })
    try {
      await api.resetSession(state.session.session_id)
      dispatch({ type: 'resetDone' })
    } catch (caught) {
      const error = toApiError(caught)
      if (error.kind === 'not_found') {
        clearSession()
        dispatch({ type: 'sessionLost', during: 'reset' })
      } else {
        dispatch({ type: 'resetFailed', error })
      }
    }
  }, [api, state.phase, state.session])

  const newConversation = useCallback(() => {
    if (state.phase === 'sending' || state.phase === 'resetting') return
    const previous = state.session
    clearSession()
    dispatch({ type: 'newConversation' })
    // Best effort: a 404 or an unreachable API is fine, the server expires
    // idle sessions on its own (API_SESSION_TTL_SECONDS).
    if (previous) void api.deleteSession(previous.session_id).catch(() => undefined)
  }, [api, state.phase, state.session])

  const dismissError = useCallback(() => dispatch({ type: 'dismissError' }), [])
  const dismissNotice = useCallback(() => dispatch({ type: 'dismissNotice' }), [])

  return { state, canSend, send, retry, stopWaiting, refresh, reset, newConversation, dismissError, dismissNotice }
}
