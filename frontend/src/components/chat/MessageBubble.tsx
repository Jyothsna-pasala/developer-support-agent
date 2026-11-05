import SmartToyIcon from '@mui/icons-material/SmartToy'
import WarningAmberIcon from '@mui/icons-material/WarningAmber'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'

import type { UiMessage } from '../../conversation/types'
import { MONOSPACE } from '../../theme'
import { isFallbackReply } from '../../utils/fallback'
import { parseToolSummary } from '../../utils/history'
import { RestoredToolCallCard, ToolCallCard } from '../tools/ToolCallCard'
import { MarkdownContent } from './MarkdownContent'

function formatTime(iso: string): string {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// An assistant turn (avatar, tool cards and reply) is capped on desktop to keep lines readable.
const ASSISTANT_MAX_WIDTH = { xs: '100%', md: 840 }
const USER_MAX_WIDTH = { xs: '88%', sm: '75%', md: '70%' }

/** Restored history tool entry: a card if the summary parses, else the text as returned. */
function RestoredToolEntry({ message }: { message: UiMessage }) {
  const summary = parseToolSummary(message.content)
  if (!summary) {
    return (
      <Typography variant="caption" color="text.secondary" sx={{ fontFamily: MONOSPACE, overflowWrap: 'anywhere' }}>
        {message.content}
      </Typography>
    )
  }
  return <RestoredToolCallCard summary={summary} />
}

/**
 * One assistant turn: the tools it ran, then the reply.
 *
 * Live replies carry `message.toolCalls`. After a refresh the tool calls are
 * separate history entries (`restoredTools`), grouped here so both look the
 * same. A turn may have tools but no reply when history was truncated.
 */
export function AssistantTurn({ message, restoredTools = [] }: { message?: UiMessage; restoredTools?: UiMessage[] }) {
  const liveTools = message?.toolCalls ?? []
  const hasTools = liveTools.length > 0 || restoredTools.length > 0
  const fallback = message ? isFallbackReply(message.content) : false

  return (
    <Stack
      direction="row"
      spacing={1}
      sx={{ alignItems: 'flex-start', maxWidth: ASSISTANT_MAX_WIDTH, minWidth: 0 }}
      data-testid={message ? 'assistant-message' : undefined}
    >
      <SmartToyIcon
        color="primary"
        sx={{ mt: 1, display: { xs: 'none', sm: 'block' }, visibility: message ? 'visible' : 'hidden' }}
      />
      <Stack spacing={1} sx={{ minWidth: 0, flex: 1 }}>
        {hasTools && (
          <Stack spacing={0.75} aria-label="Tool calls">
            {liveTools.map((call, index) => (
              <ToolCallCard key={`${call.name}-${index}`} call={call} />
            ))}
            {restoredTools.map((tool) => (
              <RestoredToolEntry key={tool.id} message={tool} />
            ))}
          </Stack>
        )}
        {message && (
          <>
            <Paper
              variant="outlined"
              sx={{
                px: { xs: 1.5, sm: 2 },
                py: 1.25,
                borderRadius: 2,
                borderTopLeftRadius: 4,
                // Short replies stay compact; long ones use the full column.
                alignSelf: 'flex-start',
                maxWidth: '100%',
                minWidth: 0,
              }}
            >
              {fallback && (
                <Tooltip title="Ollama failed during this turn, so the rule-based backend answered instead.">
                  <Chip
                    size="small"
                    color="warning"
                    variant="outlined"
                    icon={<WarningAmberIcon />}
                    label="Fallback response"
                    sx={{ mb: 1 }}
                  />
                </Tooltip>
              )}
              {/* The original reply text is rendered unchanged, including any fallback prefix. */}
              <MarkdownContent text={message.content} />
            </Paper>
            <Typography variant="caption" color="text.secondary">
              {formatTime(message.timestamp)}
            </Typography>
          </>
        )}
      </Stack>
    </Stack>
  )
}

export function MessageBubble({ message }: { message: UiMessage }) {
  if (message.role === 'divider') {
    return (
      <Divider sx={{ my: 1 }}>
        <Typography variant="caption" color="text.secondary">
          {message.content}
        </Typography>
      </Divider>
    )
  }

  if (message.role === 'tool') return <AssistantTurn restoredTools={[message]} />
  if (message.role === 'assistant') return <AssistantTurn message={message} />

  return (
    <Stack sx={{ alignItems: 'flex-end' }} data-testid="user-message">
      <Paper
        elevation={0}
        sx={{
          px: 1.75,
          py: 1,
          maxWidth: USER_MAX_WIDTH,
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          borderRadius: 2,
          borderTopRightRadius: 4,
          opacity: message.status === 'failed' ? 0.7 : 1,
        }}
      >
        <Typography sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{message.content}</Typography>
      </Paper>
      <Typography variant="caption" color={message.status === 'failed' ? 'error' : 'text.secondary'} sx={{ mt: 0.25 }}>
        {message.status === 'failed'
          ? 'Not sent'
          : message.status === 'unconfirmed'
            ? 'Sent · no reply received yet'
            : formatTime(message.timestamp)}
      </Typography>
    </Stack>
  )
}
