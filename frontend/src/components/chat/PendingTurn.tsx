import Button from '@mui/material/Button'
import CircularProgress from '@mui/material/CircularProgress'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useEffect, useState } from 'react'

export const SLOW_HINT_AFTER_SECONDS = 30

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

/** Shown while POST /messages is running. The API has no progress or streaming. */
export function PendingTurn({ since, onStopWaiting }: { since: number; onStopWaiting: () => void }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])
  const elapsed = Math.max(0, Math.floor((now - since) / 1000))

  return (
    <Paper variant="outlined" sx={{ px: 1.75, py: 1.25, alignSelf: 'flex-start', maxWidth: '92%' }} role="status">
      <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center', flexWrap: 'wrap', rowGap: 1 }}>
        <CircularProgress size={18} />
        <Typography>
          Thinking… <span aria-label="elapsed time">{formatElapsed(elapsed)}</span>
        </Typography>
        <Button size="small" onClick={onStopWaiting}>
          Stop waiting
        </Button>
      </Stack>
      {elapsed >= SLOW_HINT_AFTER_SECONDS && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          A local model on a CPU can take several minutes per answer, longer if other requests are queued.
          Stopping only stops this page from waiting; the server still finishes the turn.
        </Typography>
      )}
    </Paper>
  )
}
