import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { agentApi } from '../../api/agentApi'
import { ApiError } from '../../api/errors'
import { HEALTH } from '../../test/fixtures'
import { renderWithProviders } from '../../test/render'
import { AppLayout } from './AppLayout'

function renderLayout() {
  return renderWithProviders(
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<p>chat content</p>} />
      </Route>
    </Routes>,
  )
}

describe('AppLayout', () => {
  it('shows the mode and model from GET /health', async () => {
    vi.spyOn(agentApi, 'getHealth').mockResolvedValue(HEALTH)
    renderLayout()
    // The model part is a separate span (hidden on phones), so compare the whole chip label.
    const chipLabel = await screen.findByText(
      (_, element) => !!element?.classList.contains('MuiChip-label') && element.textContent === 'ollama · qwen3:4b',
    )
    expect(chipLabel).toBeInTheDocument()
    expect(screen.getByText('chat content')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Chat' })).toHaveAttribute('href', '/')
    expect(screen.getByRole('link', { name: 'Tools' })).toHaveAttribute('href', '/tools')
  })

  it('shows an offline banner with retry when the API is unreachable', async () => {
    const getHealth = vi
      .spyOn(agentApi, 'getHealth')
      .mockRejectedValueOnce(new ApiError('network', 'Cannot reach the API at http://127.0.0.1:8000.'))
      .mockResolvedValue({ ...HEALTH, mode: 'local', model: null })
    renderLayout()

    expect(await screen.findByText('API offline')).toBeInTheDocument()
    expect(screen.getByText('Cannot reach the API at http://127.0.0.1:8000.')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText('local')).toBeInTheDocument()
    expect(screen.queryByText('API offline')).not.toBeInTheDocument()
    expect(getHealth).toHaveBeenCalledTimes(2)
  })
})
