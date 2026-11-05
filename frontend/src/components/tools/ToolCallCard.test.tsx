import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import type { ToolCall } from '../../api/types'
import { CALC_REPLY } from '../../test/fixtures'
import { RestoredToolCallCard, ToolCallCard } from './ToolCallCard'

describe('ToolCallCard', () => {
  it('shows the actual name, arguments and result returned by the backend', async () => {
    render(<ToolCallCard call={CALC_REPLY.tool_calls[0]} />)
    // Collapsed: name, one-line arguments and status; the details are hidden.
    const summary = screen.getByRole('button', { name: /calculate_expression/ })
    expect(summary).toHaveAttribute('aria-expanded', 'false')
    expect(screen.getByText('calculate_expression')).toBeInTheDocument()
    expect(screen.getByText('{"expression":"(1200 * 0.18) + 50"}')).toBeInTheDocument()
    expect(screen.getByText('Success')).toBeInTheDocument()
    expect(screen.getByLabelText('Tool arguments')).not.toBeVisible()

    await userEvent.click(summary)
    expect(summary).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByLabelText('Tool arguments')).toHaveTextContent('"expression": "(1200 * 0.18) + 50"')
    expect(screen.getByText(/\(1200 \* 0\.18\) \+ 50 =/)).toHaveTextContent('(1200 * 0.18) + 50 = 266')

    await userEvent.click(screen.getByRole('button', { name: 'Show raw output' }))
    expect(screen.getByLabelText('Raw tool output')).toHaveTextContent('"tool": "calculate_expression"')
  })

  it('renders knowledge base and datetime results', async () => {
    const calls: ToolCall[] = [
      {
        name: 'search_knowledge_base',
        arguments: { query: 'react cors error fix' },
        ok: true,
        output: {
          tool: 'search_knowledge_base',
          ok: true,
          result: {
            query: 'react cors error fix',
            match_count: 1,
            results: [{ category: 'Frontend Troubleshooting', title: 'CORS Errors', content: 'Fix on the server.', score: 21.5 }],
          },
        },
      },
      {
        name: 'get_current_datetime',
        arguments: {},
        ok: true,
        output: {
          tool: 'get_current_datetime',
          ok: true,
          result: {
            iso_8601: '2026-11-04T12:00:00+05:30',
            date: '2026-11-04',
            time: '12:00:00',
            weekday: 'Wednesday',
            timezone: 'India Standard Time',
            utc_offset: '+0530',
          },
        },
      },
    ]
    render(
      <>
        {calls.map((call) => (
          <ToolCallCard key={call.name} call={call} />
        ))}
      </>,
    )
    await userEvent.click(screen.getByRole('button', { name: /search_knowledge_base/ }))
    expect(screen.getByText('CORS Errors')).toBeInTheDocument()
    expect(screen.getByText('Frontend Troubleshooting')).toBeInTheDocument()
    expect(screen.getByText(/1 match for/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /get_current_datetime/ }))
    expect(screen.getByText('Wednesday, 2026-11-04 at 12:00:00')).toBeInTheDocument()
  })

  it('shows tool errors and unknown shapes as returned', async () => {
    render(
      <>
        <ToolCallCard
          call={{
            name: 'calculate_expression',
            arguments: { expression: '1/0' },
            ok: false,
            output: { tool: 'calculate_expression', ok: false, error: 'Division by zero is not allowed.' },
          }}
        />
        <ToolCallCard
          call={{
            name: 'unknown',
            arguments: { _raw: 'not json' },
            ok: true,
            output: { tool: 'unknown', ok: true, result: { anything: [1, 2] } },
          }}
        />
      </>,
    )
    expect(screen.getByText('Failed')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /expression.*1\/0/ }))
    expect(screen.getByText('Error')).toBeVisible()
    expect(screen.getByRole('alert')).toHaveTextContent('Division by zero is not allowed.')

    await userEvent.click(screen.getByRole('button', { name: /_raw/ }))
    expect(screen.getByLabelText('Tool result')).toHaveTextContent('"anything"')
  })

  it('restored history entries show only what the API recorded', async () => {
    render(
      <RestoredToolCallCard
        summary={{ name: 'calculate_expression', argumentsText: '{"expression": "(1200 * 0.18) + 50"}', ok: true }}
      />,
    )
    expect(screen.getByText('calculate_expression')).toBeInTheDocument()
    expect(screen.getByText('Success')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /calculate_expression/ }))
    expect(screen.getByLabelText('Tool arguments')).toHaveTextContent('"expression": "(1200 * 0.18) + 50"')
    expect(screen.getByText(/Not available after a refresh/)).toBeVisible()
    // No result or raw output is invented for history entries.
    expect(screen.queryByText(/= 266/)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Show raw output' })).not.toBeInTheDocument()
  })

  it('restored failed entries and unparseable arguments are shown as returned', async () => {
    render(<RestoredToolCallCard summary={{ name: 'calculate_expression', argumentsText: '{bad json', ok: false }} />)
    expect(screen.getByText('Failed')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /calculate_expression/ }))
    expect(screen.getByText('Error')).toBeVisible()
    expect(screen.getByLabelText('Tool arguments')).toHaveTextContent('{bad json')
  })
})
