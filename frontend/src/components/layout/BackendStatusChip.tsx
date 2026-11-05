import CircleIcon from '@mui/icons-material/Circle'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Tooltip from '@mui/material/Tooltip'

import type { HealthState } from '../../hooks/useHealth'

/** Shows API reachability plus the mode and model reported by GET /health. */
export function BackendStatusChip({ health, error, loading }: Pick<HealthState, 'health' | 'error' | 'loading'>) {
  if (error) {
    return (
      <Tooltip title={error.message}>
        <Chip size="small" color="error" icon={<CircleIcon />} label={error.isOffline ? 'API offline' : 'API error'} />
      </Tooltip>
    )
  }
  if (!health) {
    return <Chip size="small" variant="outlined" label={loading ? 'Connecting…' : 'Unknown'} />
  }
  // The model tag is hidden on phones so the app title keeps its space.
  const label = (
    <>
      {health.mode}
      {health.model && (
        <Box component="span" sx={{ display: { xs: 'none', sm: 'inline' } }}>
          {` · ${health.model}`}
        </Box>
      )}
    </>
  )
  return (
    <Tooltip
      title={
        `API reachable. Sessions ${health.active_sessions}/${health.max_sessions}. ` +
        '/health reports the configured model; it does not check that Ollama is running.'
      }
    >
      <Chip size="small" color="success" variant="outlined" icon={<CircleIcon />} label={label} />
    </Tooltip>
  )
}
