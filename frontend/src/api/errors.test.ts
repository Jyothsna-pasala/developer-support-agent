import { AxiosError, CanceledError, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios'
import { describe, expect, it } from 'vitest'

import { ApiError, readDetail, toApiError } from './errors'

const config = { headers: {} } as InternalAxiosRequestConfig

function httpError(status: number, data: unknown): AxiosError {
  const response = { status, statusText: '', data, headers: {}, config } as AxiosResponse
  return new AxiosError(`Request failed with status code ${status}`, 'ERR_BAD_RESPONSE', config, {}, response)
}

describe('toApiError', () => {
  it('maps the documented status codes to kinds and keeps the backend detail', () => {
    const cases: [number, string, ApiError['kind']][] = [
      [404, 'Session not found. It may have expired or been deleted.', 'not_found'],
      [409, 'This session is still processing a previous request. Wait for it to finish.', 'busy'],
      [503, 'The model is busy with other requests. Try again in a moment.', 'unavailable'],
    ]
    for (const [status, detail, kind] of cases) {
      const error = toApiError(httpError(status, { detail }))
      expect(error).toBeInstanceOf(ApiError)
      expect(error.kind).toBe(kind)
      expect(error.status).toBe(status)
      expect(error.message).toBe(detail)
    }
  })

  it('reads FastAPI validation errors (detail is an array)', () => {
    const error = toApiError(
      httpError(422, {
        detail: [
          {
            type: 'value_error',
            loc: ['body', 'message'],
            msg: 'Value error, Please enter a question.',
            input: '   ',
            ctx: { error: {} },
          },
        ],
      }),
    )
    expect(error.kind).toBe('validation')
    expect(error.message).toBe('Please enter a question.')
  })

  it('reads an agent-side 422 whose detail is a string', () => {
    expect(toApiError(httpError(422, { detail: 'Please keep questions under 2000 characters.' })).message).toBe(
      'Please keep questions under 2000 characters.',
    )
  })

  it('handles a plain-text 500 body', () => {
    const error = toApiError(httpError(500, 'Internal Server Error'))
    expect(error.kind).toBe('server')
    expect(error.message).toBe('The API returned an unexpected error (HTTP 500).')
  })

  it('treats a missing response as offline', () => {
    const error = toApiError(new AxiosError('Network Error', 'ERR_NETWORK', config, {}))
    expect(error.kind).toBe('network')
    expect(error.isOffline).toBe(true)
    expect(error.message).toContain('python -m app.api')
  })

  it('recognises timeouts and cancellation', () => {
    expect(toApiError(new AxiosError('timeout', 'ECONNABORTED', config, {})).kind).toBe('timeout')
    expect(toApiError(new CanceledError()).kind).toBe('aborted')
  })

  it('passes ApiErrors through and wraps unknown values', () => {
    const original = new ApiError('busy', 'x', 409)
    expect(toApiError(original)).toBe(original)
    expect(toApiError(new Error('boom')).kind).toBe('server')
  })
})

describe('readDetail', () => {
  it('returns null for bodies without a usable detail', () => {
    expect(readDetail(null)).toBeNull()
    expect(readDetail({ other: 1 })).toBeNull()
    expect(readDetail({ detail: [] })).toBeNull()
  })
})
