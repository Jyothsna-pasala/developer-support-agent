import Box from '@mui/material/Box'
import Stack from '@mui/material/Stack'
import { useEffect, useRef, type ReactNode } from 'react'

import type { UiMessage } from '../../conversation/types'
import { AssistantTurn, MessageBubble } from './MessageBubble'

type ListItem =
  | { kind: 'message'; key: string; message: UiMessage }
  | { kind: 'assistant'; key: string; message?: UiMessage; tools: UiMessage[] }

/** Attach restored history tool entries to the assistant reply that follows them. */
function groupTurns(messages: UiMessage[]): ListItem[] {
  const items: ListItem[] = []
  let tools: UiMessage[] = []
  const flushTools = () => {
    if (tools.length) items.push({ kind: 'assistant', key: tools[0].id, tools })
    tools = []
  }
  for (const message of messages) {
    if (message.role === 'tool') {
      tools.push(message)
    } else if (message.role === 'assistant') {
      items.push({ kind: 'assistant', key: message.id, message, tools })
      tools = []
    } else {
      flushTools()
      items.push({ kind: 'message', key: message.id, message })
    }
  }
  flushTools()
  return items
}

/** The conversation, oldest first, kept scrolled to the newest entry. */
export function MessageList({ messages, footer }: { messages: UiMessage[]; footer?: ReactNode }) {
  const endRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [messages, footer])

  return (
    <Stack
      spacing={{ xs: 1.5, sm: 2 }}
      sx={{ py: 2, pl: 0.5, pr: { xs: 1.5, sm: 1 } }}
      aria-label="Conversation"
      aria-live="polite"
    >
      {groupTurns(messages).map((item) =>
        item.kind === 'assistant' ? (
          <AssistantTurn key={item.key} message={item.message} restoredTools={item.tools} />
        ) : (
          <MessageBubble key={item.key} message={item.message} />
        ),
      )}
      {/* Aligned with the assistant replies (past the avatar column). */}
      {footer && <Box sx={{ display: 'flex', pl: { xs: 0, sm: 4 } }}>{footer}</Box>}
      <div ref={endRef} />
    </Stack>
  )
}
