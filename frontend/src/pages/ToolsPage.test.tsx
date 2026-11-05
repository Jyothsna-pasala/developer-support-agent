import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { agentApi } from '../api/agentApi'
import { ApiError } from '../api/errors'
import { TOOLS } from '../test/fixtures'
import { renderWithProviders } from '../test/render'
import { ToolsPage } from './ToolsPage'

describe('ToolsPage', () => {
  it('lists the tools from GET /tools with their input schema', async () => {
    vi.spyOn(agentApi, 'getTools').mockResolvedValue(TOOLS)
    renderWithProviders(<ToolsPage />)

    expect(screen.getByLabelText('Loading tools')).toBeInTheDocument()
    expect(await screen.findByText('calculate_expression')).toBeInTheDocument()
    expect(screen.getByText('get_current_datetime')).toBeInTheDocument()
    expect(screen.getByText('expression')).toBeInTheDocument()
    expect(screen.getByText('required')).toBeInTheDocument()
    expect(screen.getByText('No arguments')).toBeInTheDocument()
    expect(screen.getByText('Object with the original expression and numeric result.')).toBeInTheDocument()
  })

  it('shows an error with retry', async () => {
    const getTools = vi
      .spyOn(agentApi, 'getTools')
      .mockRejectedValueOnce(new ApiError('network', 'Cannot reach the API.'))
      .mockResolvedValue(TOOLS)
    renderWithProviders(<ToolsPage />)

    expect(await screen.findByText('Cannot reach the API.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText('calculate_expression')).toBeInTheDocument()
    expect(getTools).toHaveBeenCalledTimes(2)
  })

  it('shows an empty state', async () => {
    vi.spyOn(agentApi, 'getTools').mockResolvedValue([])
    renderWithProviders(<ToolsPage />)
    expect(await screen.findByText('The API reports no tools.')).toBeInTheDocument()
  })
})
