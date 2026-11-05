import BuildIcon from '@mui/icons-material/Build'
import ChatIcon from '@mui/icons-material/Chat'
import SupportAgentIcon from '@mui/icons-material/SupportAgent'
import Alert from '@mui/material/Alert'
import AppBar from '@mui/material/AppBar'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Stack from '@mui/material/Stack'
import Toolbar from '@mui/material/Toolbar'
import Typography from '@mui/material/Typography'
import { NavLink, Outlet } from 'react-router-dom'

import { useHealth } from '../../hooks/useHealth'
import { BackendStatusChip } from './BackendStatusChip'

const NAV = [
  { to: '/', label: 'Chat', icon: <ChatIcon fontSize="small" /> },
  { to: '/tools', label: 'Tools', icon: <BuildIcon fontSize="small" /> },
]

export function AppLayout() {
  const { health, error, loading, refresh } = useHealth()

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100dvh' }}>
      <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Toolbar sx={{ gap: 1, px: { xs: 1.5, sm: 2 } }}>
          <SupportAgentIcon color="primary" />
          <Typography variant="h6" component="h1" noWrap sx={{ fontSize: { xs: '1rem', sm: '1.15rem' } }}>
            <Box component="span" sx={{ display: { xs: 'none', sm: 'inline' } }}>
              Developer Support Agent
            </Box>
            <Box component="span" sx={{ display: { xs: 'inline', sm: 'none' } }}>
              Dev Support
            </Box>
          </Typography>
          <Stack component="nav" direction="row" spacing={0.5} sx={{ ml: { xs: 0.5, sm: 2 } }} aria-label="Main">
            {NAV.map((item) => (
              <Button
                key={item.to}
                component={NavLink}
                to={item.to}
                end
                size="small"
                startIcon={item.icon}
                aria-label={item.label}
                sx={{
                  color: 'text.secondary',
                  '&.active': { color: 'primary.main', bgcolor: 'action.selected' },
                  minWidth: 0,
                  // Icon-only on phones.
                  '& .MuiButton-startIcon': { mr: { xs: 0, sm: 1 }, ml: { xs: 0, sm: -0.5 } },
                }}
              >
                <Box component="span" sx={{ display: { xs: 'none', sm: 'inline' } }}>
                  {item.label}
                </Box>
              </Button>
            ))}
          </Stack>
          <Box sx={{ flexGrow: 1 }} />
          <BackendStatusChip health={health} error={error} loading={loading} />
        </Toolbar>
      </AppBar>

      {error?.isOffline && (
        <Alert
          severity="error"
          square
          action={
            <Button color="inherit" size="small" onClick={refresh}>
              Retry
            </Button>
          }
        >
          {error.message}
        </Alert>
      )}

      <Box component="main" sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        <Outlet />
      </Box>
    </Box>
  )
}
