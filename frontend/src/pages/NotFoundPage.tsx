import Button from '@mui/material/Button'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <Stack spacing={2} sx={{ alignItems: 'center', textAlign: 'center', py: 8, px: 2 }}>
      <Typography variant="h5" component="h2">
        Page not found
      </Typography>
      <Button component={Link} to="/" variant="contained">
        Back to chat
      </Button>
    </Stack>
  )
}
