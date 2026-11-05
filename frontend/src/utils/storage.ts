import type { SessionResponse } from '../api/types'

export const SESSION_STORAGE_KEY = 'dsa.session'

/**
 * The active session as returned by POST /sessions. Only the identifiers are
 * stored; the conversation itself is reloaded from GET /history.
 */
export type StoredSession = SessionResponse

function isStoredSession(value: unknown): value is StoredSession {
  if (typeof value !== 'object' || value === null) return false
  const v = value as Record<string, unknown>
  return typeof v.session_id === 'string' && typeof v.backend === 'string' && typeof v.created_at === 'string'
}

// localStorage can be missing or throw (private mode, blocked storage), so
// every access is guarded and the app works without it.

export function loadSession(): StoredSession | null {
  try {
    const raw = window.localStorage.getItem(SESSION_STORAGE_KEY)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    return isStoredSession(parsed) ? parsed : null
  } catch {
    return null
  }
}

export function saveSession(session: StoredSession): void {
  try {
    window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session))
  } catch {
    // Not persisted; the session still works until the page is reloaded.
  }
}

export function clearSession(): void {
  try {
    window.localStorage.removeItem(SESSION_STORAGE_KEY)
  } catch {
    // Nothing to clear.
  }
}
