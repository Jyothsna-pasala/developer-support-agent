import { AxiosError, type InternalAxiosRequestConfig } from 'axios'
import { describe, expect, it, vi } from 'vitest'

import { CALC_REPLY, HEALTH, HISTORY, SESSION, TOOLS } from '../test/fixtures'
import { agentApi } from './agentApi'
import { apiClient, MESSAGE_TIMEOUT_MS } from './client'
import { ApiError } from './errors'

describe('agentApi', () => {
  it('calls exactly the backend endpoints with the right methods', async () => {
    const get = vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => ({
      data: url === '/health' ? HEALTH : url === '/tools' ? TOOLS : HISTORY,
    }))
    const post = vi.spyOn(apiClient, 'post').mockImplementation(async (url: string) => ({
      data: url === '/sessions' ? SESSION : url.endsWith('/messages') ? CALC_REPLY : undefined,
    }))
    const del = vi.spyOn(apiClient, 'delete').mockResolvedValue({ data: undefined })
    const id = SESSION.session_id

    expect(await agentApi.getHealth()).toEqual(HEALTH)
    expect(await agentApi.getTools()).toEqual(TOOLS)
    expect(await agentApi.createSession()).toEqual(SESSION)
    expect(await agentApi.sendMessage(id, 'Calculate (1200 * 0.18) + 50')).toEqual(CALC_REPLY)
    expect(await agentApi.getHistory(id)).toEqual(HISTORY)
    await agentApi.resetSession(id)
    await agentApi.deleteSession(id)

    expect(get.mock.calls.map((c) => c[0])).toEqual(['/health', '/tools', `/sessions/${id}/history`])
    expect(post.mock.calls.map((c) => c[0])).toEqual(['/sessions', `/sessions/${id}/messages`, `/sessions/${id}/reset`])
    expect(del.mock.calls.map((c) => c[0])).toEqual([`/sessions/${id}`])
  })

  it('sends the ChatRequest body with the long message timeout', async () => {
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: CALC_REPLY })
    const controller = new AbortController()
    await agentApi.sendMessage('abc', 'hello', controller.signal)
    expect(post).toHaveBeenCalledWith('/sessions/abc/messages', { message: 'hello' }, {
      signal: controller.signal,
      timeout: MESSAGE_TIMEOUT_MS,
    })
  })

  it('encodes the session id in the path', async () => {
    const get = vi.spyOn(apiClient, 'get').mockResolvedValue({ data: HISTORY })
    await agentApi.getHistory('a/b')
    expect(get.mock.calls[0][0]).toBe('/sessions/a%2Fb/history')
  })

  it('rethrows failures as ApiError', async () => {
    const config = { headers: {} } as InternalAxiosRequestConfig
    vi.spyOn(apiClient, 'post').mockRejectedValue(new AxiosError('Network Error', 'ERR_NETWORK', config, {}))
    await expect(agentApi.createSession()).rejects.toBeInstanceOf(ApiError)
  })
})
