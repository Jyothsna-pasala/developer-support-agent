import type { ApiError } from '../api/errors'
import type { StoredSession } from '../utils/storage'
import { nextId } from '../utils/history'
import type { UiMessage } from './types'

/**
 * - restoring: loading GET /history for the stored session
 * - idle: ready for input
 * - sending: creating a session and/or waiting for POST /messages
 * - waitingForTurn: GET /history returned 409; an earlier turn is still running
 * - resetting: POST /reset in progress
 */
export type Phase = 'restoring' | 'idle' | 'sending' | 'waitingForTurn' | 'resetting'

export interface Notice {
  severity: 'info' | 'warning'
  text: string
}

export type RetryAction = { kind: 'send'; text: string; messageId: string } | { kind: 'restore' }

export interface ConversationState {
  session: StoredSession | null
  messages: UiMessage[]
  phase: Phase
  /** Date.now() when the current turn started, for the elapsed timer. */
  pendingSince: number | null
  error: ApiError | null
  retry: RetryAction | null
  notice: Notice | null
}

export type ConversationAction =
  | { type: 'restoreStarted' }
  | { type: 'restored'; messages: UiMessage[] }
  | { type: 'restoreFailed'; error: ApiError }
  | { type: 'turnStillRunning' }
  | { type: 'sessionLost'; during: 'restore' | 'send' | 'reset'; messageId?: string; text?: string }
  | { type: 'sendStarted'; message: UiMessage; replaceId?: string; at: number }
  | { type: 'sessionCreated'; session: StoredSession }
  | { type: 'replyReceived'; userMessageId: string; reply: UiMessage }
  | { type: 'sendFailed'; error: ApiError; messageId: string; text: string }
  | { type: 'resetStarted' }
  | { type: 'resetDone' }
  | { type: 'resetFailed'; error: ApiError }
  | { type: 'newConversation' }
  | { type: 'dismissError' }
  | { type: 'dismissNotice' }

export const EXPIRED_TEXT =
  'Your previous conversation expired or the API was restarted, so it could not be restored.'

export const LOST_DURING_SEND_TEXT =
  'This conversation expired or the API was restarted, so the agent no longer has its context. ' +
  'Retry to send your message in a new conversation.'

export const STILL_RUNNING_TEXT =
  'The agent is still answering an earlier question in this conversation. ' +
  'The page will update when it finishes.'

export const STOPPED_WAITING_TEXT =
  'Stopped waiting for the reply. The server keeps working on it; use "Check for reply" in a moment.'

export function initialState(session: StoredSession | null): ConversationState {
  return {
    session,
    messages: [],
    phase: session ? 'restoring' : 'idle',
    pendingSince: null,
    error: null,
    retry: null,
    notice: null,
  }
}

function setStatus(messages: UiMessage[], id: string, status: UiMessage['status']): UiMessage[] {
  return messages.map((m) => (m.id === id ? { ...m, status } : m))
}

export function conversationReducer(state: ConversationState, action: ConversationAction): ConversationState {
  switch (action.type) {
    case 'restoreStarted':
      return { ...state, phase: 'restoring', error: null, retry: null }

    case 'restored':
      return { ...state, phase: 'idle', messages: action.messages, error: null, retry: null, notice: null }

    case 'restoreFailed':
      return { ...state, phase: 'idle', error: action.error, retry: { kind: 'restore' } }

    case 'turnStillRunning':
      return { ...state, phase: 'waitingForTurn', notice: { severity: 'info', text: STILL_RUNNING_TEXT } }

    case 'sessionLost':
      if (action.during === 'send' && action.messageId && action.text !== undefined) {
        return {
          ...state,
          session: null,
          phase: 'idle',
          pendingSince: null,
          messages: setStatus(state.messages, action.messageId, 'failed'),
          error: null,
          retry: { kind: 'send', text: action.text, messageId: action.messageId },
          notice: { severity: 'warning', text: LOST_DURING_SEND_TEXT },
        }
      }
      return {
        ...state,
        session: null,
        phase: 'idle',
        pendingSince: null,
        messages: [],
        error: null,
        retry: null,
        notice: { severity: 'warning', text: EXPIRED_TEXT },
      }

    case 'sendStarted': {
      const kept = action.replaceId ? state.messages.filter((m) => m.id !== action.replaceId) : state.messages
      return {
        ...state,
        phase: 'sending',
        pendingSince: action.at,
        messages: [...kept, action.message],
        error: null,
        retry: null,
      }
    }

    case 'sessionCreated': {
      // A conversation that was lost is still on screen: mark where the new one starts.
      const earlier = state.messages.slice(0, -1)
      const messages =
        earlier.some((m) => m.role !== 'divider')
          ? [
              ...earlier,
              {
                id: nextId('d'),
                role: 'divider' as const,
                content: 'New conversation',
                timestamp: new Date().toISOString(),
                origin: 'live' as const,
              },
              ...state.messages.slice(-1),
            ]
          : state.messages
      return { ...state, session: action.session, messages, notice: null }
    }

    case 'replyReceived':
      return {
        ...state,
        phase: 'idle',
        pendingSince: null,
        messages: [...setStatus(state.messages, action.userMessageId, undefined), action.reply],
        error: null,
        retry: null,
      }

    case 'sendFailed': {
      const base = { ...state, phase: 'idle' as const, pendingSince: null }
      if (action.error.kind === 'aborted' || action.error.kind === 'timeout') {
        // The server may still finish the turn, so resending could duplicate it.
        return {
          ...base,
          messages: setStatus(state.messages, action.messageId, 'unconfirmed'),
          notice: {
            severity: 'info',
            text: action.error.kind === 'aborted' ? STOPPED_WAITING_TEXT : `${action.error.message} ${STOPPED_WAITING_TEXT}`,
          },
        }
      }
      return {
        ...base,
        messages: setStatus(state.messages, action.messageId, 'failed'),
        error: action.error,
        retry: { kind: 'send', text: action.text, messageId: action.messageId },
      }
    }

    case 'resetStarted':
      return { ...state, phase: 'resetting', error: null, retry: null }

    case 'resetDone':
      return { ...state, phase: 'idle', messages: [], notice: null }

    case 'resetFailed':
      return { ...state, phase: 'idle', error: action.error }

    case 'newConversation':
      return initialState(null)

    case 'dismissError':
      return { ...state, error: null, retry: null }

    case 'dismissNotice':
      return { ...state, notice: null }
  }
}
