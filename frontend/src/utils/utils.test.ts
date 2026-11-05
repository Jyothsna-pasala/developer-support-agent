import { describe, expect, it } from 'vitest'

import { HISTORY, SESSION } from '../test/fixtures'
import { FALLBACK_PREFIX, isFallbackReply } from './fallback'
import { historyToUiMessages, parseToolSummary } from './history'
import { clearSession, loadSession, saveSession, SESSION_STORAGE_KEY } from './storage'
import { isCalculationResult, isDateTimeResult, isKnowledgeBaseResult } from './toolResults'
import { countChars, MAX_MESSAGE_CHARS, validateMessage } from './validation'

describe('validateMessage', () => {
  it('uses the backend rules and messages', () => {
    expect(validateMessage('')).toBe('Please enter a question.')
    expect(validateMessage('   \n ')).toBe('Please enter a question.')
    expect(validateMessage('  hi  ')).toBeNull()
    expect(validateMessage('x'.repeat(MAX_MESSAGE_CHARS))).toBeNull()
    expect(validateMessage('x'.repeat(MAX_MESSAGE_CHARS + 1))).toBe('Please keep questions under 2000 characters.')
  })

  it('counts code points like Python, not UTF-16 units', () => {
    expect('😀'.length).toBe(2)
    expect(countChars('😀')).toBe(1)
    expect(validateMessage('😀'.repeat(MAX_MESSAGE_CHARS))).toBeNull()
  })
})

describe('isFallbackReply', () => {
  it('detects the FallbackBackend prefix only at the start', () => {
    expect(isFallbackReply(`${FALLBACK_PREFIX} Cannot reach Ollama. Answered by the rule-based fallback.]\nhi`)).toBe(true)
    expect(isFallbackReply('The text [LLM unavailable: is quoted]')).toBe(false)
    expect(isFallbackReply('Normal answer')).toBe(false)
  })
})

describe('history mapping', () => {
  it('keeps roles, content and server timestamps in order', () => {
    const messages = historyToUiMessages(HISTORY.messages)
    expect(messages.map((m) => m.role)).toEqual(['user', 'tool', 'assistant'])
    expect(messages.every((m) => m.origin === 'history')).toBe(true)
    expect(messages[0].timestamp).toBe(HISTORY.messages[0].timestamp)
    expect(new Set(messages.map((m) => m.id)).size).toBe(3)
  })

  it('parses tool summary entries from DeveloperSupportAgent.ask()', () => {
    expect(parseToolSummary('calculate_expression({"expression": "(1200 * 0.18) + 50"}) -> ok')).toEqual({
      name: 'calculate_expression',
      argumentsText: '{"expression": "(1200 * 0.18) + 50"}',
      ok: true,
    })
    expect(parseToolSummary('get_current_datetime({}) -> error')).toEqual({
      name: 'get_current_datetime',
      argumentsText: '{}',
      ok: false,
    })
    expect(parseToolSummary('something else')).toBeNull()
  })
})

describe('session storage', () => {
  it('saves, loads and clears the session', () => {
    expect(loadSession()).toBeNull()
    saveSession(SESSION)
    expect(loadSession()).toEqual(SESSION)
    clearSession()
    expect(loadSession()).toBeNull()
  })

  it('ignores corrupt or foreign values', () => {
    window.localStorage.setItem(SESSION_STORAGE_KEY, '{not json')
    expect(loadSession()).toBeNull()
    window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({ session_id: 1 }))
    expect(loadSession()).toBeNull()
  })
})

describe('tool result guards', () => {
  it('accept the backend shapes and reject others', () => {
    expect(isCalculationResult({ expression: '6 * 7', result: 42 })).toBe(true)
    expect(isCalculationResult({ expression: '6 * 7' })).toBe(false)
    expect(
      isDateTimeResult({
        iso_8601: '2026-11-04T12:00:00+05:30',
        date: '2026-11-04',
        time: '12:00:00',
        weekday: 'Wednesday',
        timezone: 'India Standard Time',
        utc_offset: '+0530',
      }),
    ).toBe(true)
    expect(isDateTimeResult({ date: '2026-11-04' })).toBe(false)
    expect(
      isKnowledgeBaseResult({
        query: 'cors',
        match_count: 1,
        results: [{ category: 'Frontend', title: 'CORS Errors', content: '...', score: 9 }],
      }),
    ).toBe(true)
    expect(isKnowledgeBaseResult({ query: 'cors', match_count: 1, results: [{}] })).toBe(false)
  })
})
