import { ThemeProvider } from '@mui/material/styles'
import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'

import type { AgentApi } from '../api/agentApi'
import { ConversationProvider } from '../conversation/ConversationProvider'
import { theme } from '../theme'

export function renderWithProviders(ui: ReactElement, { api, route = '/' }: { api?: AgentApi; route?: string } = {}) {
  return render(
    <ThemeProvider theme={theme}>
      <MemoryRouter initialEntries={[route]}>
        <ConversationProvider api={api}>{ui}</ConversationProvider>
      </MemoryRouter>
    </ThemeProvider>,
  )
}
