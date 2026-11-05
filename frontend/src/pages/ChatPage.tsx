import AddCommentIcon from '@mui/icons-material/AddComment'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Container from '@mui/material/Container'
import LinearProgress from '@mui/material/LinearProgress'
import Skeleton from '@mui/material/Skeleton'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useState } from 'react'

import { ChatInput } from '../components/chat/ChatInput'
import { EmptyConversation } from '../components/chat/EmptyConversation'
import { MessageList } from '../components/chat/MessageList'
import { PendingTurn } from '../components/chat/PendingTurn'
import { ConfirmDialog } from '../components/feedback/ConfirmDialog'
import { ErrorAlert } from '../components/feedback/ErrorAlert'
import { useConversationContext } from '../conversation/context'
import type { Phase } from '../conversation/reducer'

const DISABLED_REASON: Record<Phase, string | undefined> = {
  idle: undefined,
  restoring: 'Loading the conversation…',
  sending: 'Waiting for the reply…',
  waitingForTurn: 'Waiting for an earlier answer to finish…',
  resetting: 'Resetting…',
}

function RestoringSkeleton() {
  return (
    <Stack spacing={2} sx={{ py: 2 }} aria-label="Loading conversation">
      <Skeleton variant="rounded" height={40} width="45%" sx={{ alignSelf: 'flex-end' }} />
      <Skeleton variant="rounded" height={88} width="75%" />
      <Skeleton variant="rounded" height={40} width="35%" sx={{ alignSelf: 'flex-end' }} />
    </Stack>
  )
}

export function ChatPage() {
  const conversation = useConversationContext()
  const { state } = conversation
  const [confirm, setConfirm] = useState<'reset' | 'new' | null>(null)

  const busy = state.phase === 'sending' || state.phase === 'resetting'
  const hasUnconfirmed = state.messages.some((m) => m.status === 'unconfirmed')
  const isEmpty = state.messages.length === 0 && state.phase === 'idle'

  return (
    <Container
      maxWidth={false}
      // Wider than MUI's "md" (900 px) so tool cards and code have room; reply
      // text itself is capped in AssistantTurn to keep lines readable.
      sx={{ height: '100%', display: 'flex', flexDirection: 'column', maxWidth: 1040, px: { xs: 1.5, sm: 3 } }}
    >
      {/* Session header */}
      <Stack
        direction="row"
        spacing={1}
        sx={{ alignItems: 'center', py: 1.25, borderBottom: 1, borderColor: 'divider', flexWrap: 'wrap', rowGap: 1 }}
      >
        <Box sx={{ minWidth: 0, flex: 1 }}>
          {state.session ? (
            <Stack direction="row" spacing={1} sx={{ alignItems: 'center', minWidth: 0 }}>
              <Chip size="small" label={state.session.backend} sx={{ maxWidth: '100%' }} />
              <Typography variant="caption" color="text.secondary" noWrap sx={{ display: { xs: 'none', sm: 'block' } }}>
                Started {new Date(state.session.created_at).toLocaleString()}
              </Typography>
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">
              New conversation
            </Typography>
          )}
        </Box>
        <Button
          size="small"
          startIcon={<RestartAltIcon />}
          onClick={() => setConfirm('reset')}
          disabled={!state.session || state.phase !== 'idle'}
          aria-label="Reset conversation"
        >
          <Box component="span" sx={{ display: { xs: 'none', sm: 'inline' } }}>
            Reset
          </Box>
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<AddCommentIcon />}
          onClick={() => (state.messages.length ? setConfirm('new') : conversation.newConversation())}
          disabled={busy}
          aria-label="New conversation"
        >
          <Box component="span" sx={{ display: { xs: 'none', sm: 'inline' } }}>
            New conversation
          </Box>
        </Button>
      </Stack>

      {/* Notices and errors */}
      <Stack spacing={1} sx={{ pt: 1 }}>
        {state.phase === 'waitingForTurn' && <LinearProgress aria-label="Waiting for an earlier answer" />}
        {state.notice && (
          <Alert
            severity={state.notice.severity}
            onClose={state.phase === 'waitingForTurn' ? undefined : conversation.dismissNotice}
            action={
              hasUnconfirmed && state.phase === 'idle' ? (
                <Button color="inherit" size="small" onClick={conversation.refresh}>
                  Check for reply
                </Button>
              ) : state.retry && !state.error ? (
                <Button color="inherit" size="small" onClick={conversation.retry}>
                  Retry
                </Button>
              ) : undefined
            }
          >
            {state.notice.text}
          </Alert>
        )}
        {state.error && (
          <ErrorAlert
            error={state.error}
            onRetry={state.retry ? conversation.retry : undefined}
            retryLabel={state.retry?.kind === 'restore' ? 'Retry loading' : 'Retry'}
            onClose={conversation.dismissError}
          />
        )}
      </Stack>

      {/* Conversation */}
      <Box sx={{ flex: 1, minHeight: 0, overflowY: 'auto', overflowX: 'hidden' }}>
        {state.phase === 'restoring' ? (
          <RestoringSkeleton />
        ) : isEmpty ? (
          <EmptyConversation onPick={conversation.send} disabled={!conversation.canSend} />
        ) : (
          <MessageList
            messages={state.messages}
            footer={
              state.phase === 'sending' && state.pendingSince !== null ? (
                <PendingTurn since={state.pendingSince} onStopWaiting={conversation.stopWaiting} />
              ) : null
            }
          />
        )}
      </Box>

      {/* Input */}
      <Box sx={{ py: 1.5, borderTop: 1, borderColor: 'divider' }}>
        <ChatInput
          onSend={conversation.send}
          disabled={!conversation.canSend}
          disabledReason={DISABLED_REASON[state.phase]}
        />
      </Box>

      <ConfirmDialog
        open={confirm === 'reset'}
        title="Reset this conversation?"
        text="The transcript and the agent's memory of this conversation are cleared. The session stays open."
        confirmLabel="Reset"
        onConfirm={() => void conversation.reset()}
        onClose={() => setConfirm(null)}
      />
      <ConfirmDialog
        open={confirm === 'new'}
        title="Start a new conversation?"
        text="The current conversation is deleted on the server and cannot be restored."
        confirmLabel="New conversation"
        onConfirm={conversation.newConversation}
        onClose={() => setConfirm(null)}
      />
    </Container>
  )
}
