import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Button from '@mui/material/Button'
import Stack from '@mui/material/Stack'

import type { ApiError, ApiErrorKind } from '../../api/errors'

const TITLES: Record<ApiErrorKind, string> = {
  network: 'Cannot reach the API',
  timeout: 'Request timed out',
  aborted: 'Request cancelled',
  not_found: 'Conversation not found',
  busy: 'Conversation busy',
  validation: 'Message rejected',
  unavailable: 'Service unavailable',
  server: 'Unexpected error',
}

const HINTS: Partial<Record<ApiErrorKind, string>> = {
  busy: 'Another tab or request is still using this conversation. Wait for it to finish, then retry.',
  unavailable: 'The local model handles one request at a time. Retry in about 10 seconds.',
}

interface ErrorAlertProps {
  error: ApiError
  onRetry?: () => void
  retryLabel?: string
  onClose?: () => void
}

/** Displays an ApiError with a title per kind and an optional Retry action. */
export function ErrorAlert({ error, onRetry, retryLabel = 'Retry', onClose }: ErrorAlertProps) {
  const hint = HINTS[error.kind]
  return (
    <Alert
      severity={error.kind === 'unavailable' || error.kind === 'busy' ? 'warning' : 'error'}
      onClose={onClose}
      action={
        onRetry && (
          <Stack direction="row">
            <Button color="inherit" size="small" onClick={onRetry}>
              {retryLabel}
            </Button>
          </Stack>
        )
      }
    >
      <AlertTitle>
        {TITLES[error.kind]}
        {error.status ? ` (HTTP ${error.status})` : ''}
      </AlertTitle>
      {error.message}
      {hint && <div>{hint}</div>}
    </Alert>
  )
}
