import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { UiMessage } from '../../conversation/types'
import { CALC_REPLY } from '../../test/fixtures'
import { MessageList } from './MessageList'

const at = '2026-11-04T12:20:14+05:30'
const msg = (id: string, role: UiMessage['role'], content: string, extra: Partial<UiMessage> = {}): UiMessage => ({
  id,
  role,
  content,
  timestamp: at,
  origin: 'history',
  ...extra,
})

describe('MessageList', () => {
  it('groups restored tool entries with the reply that follows them', () => {
    render(
      <MessageList
        messages={[
          msg('u1', 'user', 'What time is it and what is 2+2?'),
          msg('t1', 'tool', 'get_current_datetime({}) -> ok'),
          msg('t2', 'tool', 'calculate_expression({"expression": "2/0"}) -> error'),
          msg('a1', 'assistant', 'It is noon. 2/0 is undefined.'),
          msg('u2', 'user', 'thanks'),
          // History is capped at 50 entries, so a turn may end without a reply.
          msg('t3', 'tool', 'search_knowledge_base({"query": "cors"}) -> ok'),
        ]}
      />,
    )

    const [assistant] = screen.getAllByTestId('assistant-message')
    const tools = within(assistant).getByLabelText('Tool calls')
    const cards = within(tools).getAllByRole('button')
    expect(cards.map((c) => c.textContent)).toEqual([
      'get_current_datetime{}Success',
      'calculate_expression{"expression": "2/0"}Failed',
    ])
    expect(within(assistant).getByText('It is noon. 2/0 is undefined.')).toBeInTheDocument()

    // The trailing tool entry is still shown, with no reply and no assistant test id.
    expect(screen.getAllByTestId('assistant-message')).toHaveLength(1)
    expect(screen.getByRole('button', { name: /search_knowledge_base/ })).toBeInTheDocument()
    expect(screen.getAllByLabelText('Tool calls')).toHaveLength(2)
  })

  it('shows live tool calls before the reply in the same turn', () => {
    render(
      <MessageList
        messages={[msg('a1', 'assistant', CALC_REPLY.reply, { origin: 'live', toolCalls: CALC_REPLY.tool_calls })]}
      />,
    )
    const assistant = screen.getByTestId('assistant-message')
    const tools = within(assistant).getByLabelText('Tool calls')
    expect(within(tools).getByRole('button', { name: /calculate_expression/ })).toHaveTextContent('Success')
    // Tool cards come before the reply text.
    const reply = assistant.querySelector('.markdown')!
    expect(reply).toHaveTextContent('The result of (1200 * 0.18) + 50 is 266.')
    expect(tools.compareDocumentPosition(reply) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('keeps an unrecognised history tool entry as plain text', () => {
    render(<MessageList messages={[msg('t1', 'tool', 'something unexpected'), msg('a1', 'assistant', 'ok')]} />)
    expect(within(screen.getByLabelText('Tool calls')).getByText('something unexpected')).toBeInTheDocument()
  })
})
