import SendIcon from '@mui/icons-material/Send'
import IconButton from '@mui/material/IconButton'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import { useState, type FormEvent, type KeyboardEvent } from 'react'

import { countChars, MAX_MESSAGE_CHARS, validateMessage } from '../../utils/validation'

interface ChatInputProps {
  /** Returns false if the message was not accepted. */
  onSend: (text: string) => boolean
  disabled: boolean
  disabledReason?: string
}

export function ChatInput({ onSend, disabled, disabledReason }: ChatInputProps) {
  const [draft, setDraft] = useState('')
  const [touched, setTouched] = useState(false)
  const length = countChars(draft.trim())
  const validation = validateMessage(draft)
  const tooLong = length > MAX_MESSAGE_CHARS

  const submit = (event?: FormEvent) => {
    event?.preventDefault()
    setTouched(true)
    if (disabled || validation) return
    if (onSend(draft)) {
      setDraft('')
      setTouched(false)
    }
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    // Enter sends, Shift+Enter adds a line. Ignore Enter while an IME is composing.
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  const showError = (touched && validation !== null && draft.length > 0) || tooLong
  const helper = showError ? validation : disabled ? disabledReason : 'Enter to send · Shift+Enter for a new line'

  return (
    <Stack component="form" onSubmit={submit} direction="row" spacing={1} sx={{ alignItems: 'flex-start' }} noValidate>
      <TextField
        fullWidth
        multiline
        maxRows={6}
        size="small"
        placeholder="Ask about Python, React, TypeScript, APIs… or try “calculate 2^10”"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={onKeyDown}
        error={showError}
        helperText={
          <Stack component="span" direction="row" sx={{ justifyContent: 'space-between', gap: 1 }}>
            <span>{helper}</span>
            <Typography component="span" variant="caption" color={tooLong ? 'error' : 'text.secondary'}>
              {length}/{MAX_MESSAGE_CHARS}
            </Typography>
          </Stack>
        }
        slotProps={{ htmlInput: { 'aria-label': 'Message' } }}
      />
      <Tooltip title={disabled ? (disabledReason ?? '') : 'Send'}>
        <span>
          <IconButton
            type="submit"
            color="primary"
            aria-label="Send message"
            disabled={disabled || validation !== null}
            sx={{ mt: 0.25 }}
          >
            <SendIcon />
          </IconButton>
        </span>
      </Tooltip>
    </Stack>
  )
}
