import { act, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../api/errors'
import type { ChatResponse } from '../api/types'
import { BUSY_POLL_MS } from '../conversation/useConversation'
import { CALC_REPLY, createFakeApi, deferred, FALLBACK_REPLY, HISTORY, SESSION } from '../test/fixtures'
import { renderWithProviders } from '../test/render'
import { loadSession, saveSession } from '../utils/storage'
import { ChatPage } from './ChatPage'

const input = () => screen.getByRole('textbox', { name: 'Message' })

async function type(text: string) {
  await userEvent.type(input(), text)
}

afterEach(() => {
  vi.useRealTimers()
})

describe('ChatPage: new conversation', () => {
  it('shows the empty state and creates no session until the first message', () => {
    const api = createFakeApi()
    renderWithProviders(<ChatPage />, { api })
    expect(screen.getByText('Ask a development question')).toBeInTheDocument()
    expect(screen.getByText('calculate (1200 * 0.18) + 50')).toBeInTheDocument()
    expect(api.createSession).not.toHaveBeenCalled()
    expect(api.getHistory).not.toHaveBeenCalled()
  })

  it('creates a session, sends the message and renders the reply with its tool call', async () => {
    const reply = deferred<ChatResponse>()
    const api = createFakeApi({ sendMessage: vi.fn().mockReturnValue(reply.promise) })
    renderWithProviders(<ChatPage />, { api })

    await type('Calculate (1200 * 0.18) + 50{Enter}')

    expect(await screen.findByText('Calculate (1200 * 0.18) + 50')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Thinking…')
    expect(input()).toHaveValue('')
    await waitFor(() => expect(api.sendMessage).toHaveBeenCalled())
    expect(api.createSession).toHaveBeenCalledTimes(1)
    expect(api.sendMessage.mock.calls[0].slice(0, 2)).toEqual([SESSION.session_id, 'Calculate (1200 * 0.18) + 50'])
    expect(loadSession()).toEqual(SESSION)

    await act(async () => reply.resolve(CALC_REPLY))

    const assistant = await screen.findByTestId('assistant-message')
    // The reply is rendered as Markdown: **266** becomes <strong>.
    expect(assistant.querySelector('.markdown strong')).toHaveTextContent('266')
    expect(within(assistant).getByText('calculate_expression')).toBeInTheDocument()
    expect(within(assistant).getByText('{"expression":"(1200 * 0.18) + 50"}')).toBeInTheDocument()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.getByText(SESSION.backend)).toBeInTheDocument()
  })

  it('prevents duplicate submissions while a turn is running', async () => {
    const reply = deferred<ChatResponse>()
    const api = createFakeApi({ sendMessage: vi.fn().mockReturnValue(reply.promise) })
    renderWithProviders(<ChatPage />, { api })

    await type('first{Enter}')
    await waitFor(() => expect(api.sendMessage).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
    expect(screen.getByText('Waiting for the reply…')).toBeInTheDocument()

    // The next question can be typed while waiting, but Enter, the send
    // button and the example prompts do not submit it.
    await userEvent.type(input(), 'second{Enter}')
    // Click through the disabled state on purpose (user-event blocks it by default).
    await userEvent.setup({ pointerEventsCheck: 0 }).click(screen.getByRole('button', { name: 'Send message' }))
    expect(api.sendMessage).toHaveBeenCalledTimes(1)
    expect(api.createSession).toHaveBeenCalledTimes(1)
    expect(input()).toHaveValue('second')
    expect(screen.getAllByTestId('user-message')).toHaveLength(1)

    await act(async () => reply.resolve(CALC_REPLY))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Send message' })).not.toBeDisabled())
    await userEvent.type(input(), '{Enter}')
    await waitFor(() => expect(api.sendMessage).toHaveBeenCalledTimes(2))
    expect(api.sendMessage.mock.calls[1][1]).toBe('second')
  })

  it('ignores a second rapid submit before React re-renders', async () => {
    const api = createFakeApi({ sendMessage: vi.fn().mockReturnValue(new Promise(() => undefined)) })
    renderWithProviders(<ChatPage />, { api })
    await type('hello')
    const form = input().closest('form')!
    // Two submit events in the same tick: only the synchronous guard can stop the second.
    act(() => {
      form.requestSubmit()
      form.requestSubmit()
    })
    await waitFor(() => expect(api.createSession).toHaveBeenCalledTimes(1))
    expect(screen.getAllByTestId('user-message')).toHaveLength(1)
  })

  it('marks fallback replies with a chip without changing the text', async () => {
    const api = createFakeApi({ sendMessage: vi.fn().mockResolvedValue(FALLBACK_REPLY) })
    renderWithProviders(<ChatPage />, { api })
    await type('calculate 6 * 7{Enter}')

    const assistant = await screen.findByTestId('assistant-message')
    expect(within(assistant).getByText('Fallback response')).toBeInTheDocument()
    expect(assistant).toHaveTextContent('[LLM unavailable: Ollama returned HTTP 500: simulated failure')
    expect(assistant).toHaveTextContent('6 * 7 = 42')
  })

  it('does not mark normal replies as fallback', async () => {
    renderWithProviders(<ChatPage />, { api: createFakeApi() })
    await type('hi{Enter}')
    await screen.findByTestId('assistant-message')
    expect(screen.queryByText('Fallback response')).not.toBeInTheDocument()
  })

  it('blocks messages over 2000 characters', async () => {
    const api = createFakeApi()
    renderWithProviders(<ChatPage />, { api })
    await userEvent.click(input())
    await userEvent.paste('x'.repeat(2001))
    expect(screen.getByText('2001/2000')).toBeInTheDocument()
    expect(screen.getByText('Please keep questions under 2000 characters.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
    await userEvent.keyboard('{Enter}')
    expect(api.createSession).not.toHaveBeenCalled()
  })
})

describe('ChatPage: errors', () => {
  it('shows a 503 with Retry, and Retry resends the same message', async () => {
    const busy = new ApiError('unavailable', 'The model is busy with other requests. Try again in a moment.', 503)
    const api = createFakeApi({ sendMessage: vi.fn().mockRejectedValueOnce(busy).mockResolvedValueOnce(CALC_REPLY) })
    renderWithProviders(<ChatPage />, { api })
    await type('Calculate (1200 * 0.18) + 50{Enter}')

    expect(await screen.findByText('Service unavailable (HTTP 503)')).toBeInTheDocument()
    expect(screen.getByText(busy.message)).toBeInTheDocument()
    expect(screen.getByText('Not sent')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await screen.findByTestId('assistant-message')
    expect(api.sendMessage).toHaveBeenCalledTimes(2)
    expect(screen.getAllByTestId('user-message')).toHaveLength(1)
    expect(screen.queryByText('Not sent')).not.toBeInTheDocument()
  })

  it('shows 409 as a busy conversation', async () => {
    const conflict = new ApiError(
      'busy',
      'This session is still processing a previous request. Wait for it to finish.',
      409,
    )
    renderWithProviders(<ChatPage />, { api: createFakeApi({ sendMessage: vi.fn().mockRejectedValue(conflict) }) })
    await type('hello{Enter}')
    expect(await screen.findByText('Conversation busy (HTTP 409)')).toBeInTheDocument()
    expect(screen.getByText(/Another tab or request/)).toBeInTheDocument()
  })

  it('shows network errors', async () => {
    const offline = new ApiError('network', 'Cannot reach the API at http://127.0.0.1:8000.')
    renderWithProviders(<ChatPage />, { api: createFakeApi({ createSession: vi.fn().mockRejectedValue(offline) }) })
    await type('hello{Enter}')
    expect(await screen.findByText('Cannot reach the API')).toBeInTheDocument()
    expect(screen.getByText(offline.message)).toBeInTheDocument()
  })

  it('handles a session that expired mid-conversation (404)', async () => {
    saveSession(SESSION)
    const expired = new ApiError('not_found', 'Session not found. It may have expired or been deleted.', 404)
    const newSession = { ...SESSION, session_id: 'new-session' }
    const api = createFakeApi({
      sendMessage: vi.fn().mockRejectedValueOnce(expired).mockResolvedValueOnce(CALC_REPLY),
      createSession: vi.fn().mockResolvedValue(newSession),
    })
    renderWithProviders(<ChatPage />, { api })
    await screen.findByTestId('assistant-message') // restored history
    await type('again{Enter}')

    expect(await screen.findByText(/no longer has its context/)).toBeInTheDocument()
    expect(loadSession()).toBeNull()

    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(api.sendMessage).toHaveBeenCalledTimes(2))
    expect(api.sendMessage.mock.calls[1].slice(0, 2)).toEqual(['new-session', 'again'])
    // A divider marks where the new server-side conversation starts.
    expect(await screen.findByRole('separator')).toHaveTextContent('New conversation')
    expect(loadSession()).toEqual(newSession)
  })

  it('lets the user stop waiting, then check for the reply from history', async () => {
    const api = createFakeApi({
      sendMessage: vi.fn(
        (_id: string, _text: string, signal?: AbortSignal) =>
          new Promise<ChatResponse>((_resolve, reject) => {
            signal?.addEventListener('abort', () => reject(new ApiError('aborted', 'The request was cancelled.')))
          }),
      ),
    })
    renderWithProviders(<ChatPage />, { api })
    await type('Calculate (1200 * 0.18) + 50{Enter}')
    await userEvent.click(await screen.findByRole('button', { name: 'Stop waiting' }))

    expect(await screen.findByText(/Stopped waiting for the reply/)).toBeInTheDocument()
    expect(screen.getByText('Sent · no reply received yet')).toBeInTheDocument()
    expect(input()).not.toBeDisabled()

    await userEvent.click(screen.getByRole('button', { name: 'Check for reply' }))
    expect(api.getHistory).toHaveBeenCalledWith(SESSION.session_id, undefined)
    expect(await screen.findByTestId('assistant-message')).toHaveTextContent('266')
  })
})

describe('ChatPage: refresh and session actions', () => {
  it('restores the stored session from GET /history after a refresh', async () => {
    saveSession(SESSION)
    const api = createFakeApi()
    renderWithProviders(<ChatPage />, { api })

    expect(screen.getByLabelText('Loading conversation')).toBeInTheDocument()
    const assistant = await screen.findByTestId('assistant-message')
    expect(assistant).toHaveTextContent('266')
    expect(screen.getByText('Calculate (1200 * 0.18) + 50')).toBeInTheDocument()
    // History tool entries are summaries: shown as a card inside the reply's turn,
    // with the recorded name, arguments and status but no invented result.
    const tools = within(assistant).getByLabelText('Tool calls')
    expect(within(tools).getByText('calculate_expression')).toBeInTheDocument()
    expect(within(tools).getByText('{"expression": "(1200 * 0.18) + 50"}')).toBeInTheDocument()
    expect(within(tools).getByText('Success')).toBeInTheDocument()
    await userEvent.click(within(tools).getByRole('button', { name: /calculate_expression/ }))
    expect(within(tools).getByText(/Not available after a refresh/)).toBeVisible()
    expect(api.getHistory).toHaveBeenCalledWith(SESSION.session_id, expect.any(AbortSignal))
    expect(api.createSession).not.toHaveBeenCalled()

    await type('next{Enter}')
    await waitFor(() => expect(api.sendMessage).toHaveBeenCalled())
    expect(api.sendMessage.mock.calls[0][0]).toBe(SESSION.session_id)
  })

  it('explains an expired session on load (404) and starts fresh', async () => {
    saveSession(SESSION)
    const api = createFakeApi({
      getHistory: vi.fn().mockRejectedValue(new ApiError('not_found', 'Session not found.', 404)),
    })
    renderWithProviders(<ChatPage />, { api })
    expect(await screen.findByText(/expired or the API was restarted/)).toBeInTheDocument()
    expect(screen.getByText('Ask a development question')).toBeInTheDocument()
    expect(loadSession()).toBeNull()
  })

  it('waits while an earlier turn is still running (409), polling history', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    saveSession(SESSION)
    const getHistory = vi
      .fn()
      .mockRejectedValueOnce(new ApiError('busy', 'still processing', 409))
      .mockResolvedValue(HISTORY)
    renderWithProviders(<ChatPage />, { api: createFakeApi({ getHistory }) })

    expect(await screen.findByText(/still answering an earlier question/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
    expect(screen.getByLabelText('Waiting for an earlier answer')).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(BUSY_POLL_MS)
    })
    expect(await screen.findByTestId('assistant-message')).toHaveTextContent('266')
    expect(getHistory.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(screen.queryByText(/still answering an earlier question/)).not.toBeInTheDocument()
  })

  it('offers a retry when history cannot be loaded', async () => {
    saveSession(SESSION)
    const getHistory = vi
      .fn()
      .mockRejectedValueOnce(new ApiError('network', 'Cannot reach the API.'))
      .mockResolvedValue(HISTORY)
    renderWithProviders(<ChatPage />, { api: createFakeApi({ getHistory }) })
    await userEvent.click(await screen.findByRole('button', { name: 'Retry loading' }))
    expect(await screen.findByTestId('assistant-message')).toBeInTheDocument()
  })

  it('resets the conversation after confirmation', async () => {
    saveSession(SESSION)
    const api = createFakeApi()
    renderWithProviders(<ChatPage />, { api })
    await screen.findByTestId('assistant-message')

    await userEvent.click(screen.getByRole('button', { name: 'Reset conversation' }))
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Reset' }))

    expect(api.resetSession).toHaveBeenCalledWith(SESSION.session_id)
    expect(await screen.findByText('Ask a development question')).toBeInTheDocument()
    expect(loadSession()).toEqual(SESSION) // Same session, cleared on the server.
  })

  it('starts a new conversation by deleting the old session', async () => {
    saveSession(SESSION)
    const api = createFakeApi()
    renderWithProviders(<ChatPage />, { api })
    await screen.findByTestId('assistant-message')

    await userEvent.click(screen.getByRole('button', { name: 'New conversation' }))
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'New conversation' }))

    expect(api.deleteSession).toHaveBeenCalledWith(SESSION.session_id)
    expect(loadSession()).toBeNull()
    expect(await screen.findByText('Ask a development question')).toBeInTheDocument()
  })
})
