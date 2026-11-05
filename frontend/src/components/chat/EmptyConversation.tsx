import SupportAgentIcon from '@mui/icons-material/SupportAgent'
import Chip from '@mui/material/Chip'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'

// The same examples as the CLI's help text (app/agent.py HELP_TEXT).
const EXAMPLE_PROMPTS = [
  'What is useEffect in React?',
  'How do I fix a CORS error?',
  'calculate (1200 * 0.18) + 50',
  'What time is it?',
]

export function EmptyConversation({ onPick, disabled }: { onPick: (text: string) => void; disabled: boolean }) {
  return (
    <Stack spacing={2} sx={{ alignItems: 'center', textAlign: 'center', py: { xs: 4, sm: 8 }, px: 2 }}>
      <SupportAgentIcon color="primary" sx={{ fontSize: 48 }} />
      <Typography variant="h5" component="h2">
        Ask a development question
      </Typography>
      <Typography color="text.secondary" sx={{ maxWidth: 520 }}>
        Python, React, TypeScript, REST APIs, troubleshooting and documentation. The agent can search its
        knowledge base, calculate, and tell the date and time.
      </Typography>
      <Stack direction="row" sx={{ flexWrap: 'wrap', justifyContent: 'center', gap: 1 }}>
        {EXAMPLE_PROMPTS.map((prompt) => (
          <Chip key={prompt} label={prompt} onClick={() => onPick(prompt)} disabled={disabled} variant="outlined" />
        ))}
      </Stack>
    </Stack>
  )
}
