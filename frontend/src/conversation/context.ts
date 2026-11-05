import { createContext, useContext } from 'react'

import type { Conversation } from './useConversation'

export const ConversationContext = createContext<Conversation | null>(null)

/** The app-wide conversation; kept above the router so navigation never loses it. */
export function useConversationContext(): Conversation {
  const conversation = useContext(ConversationContext)
  if (!conversation) throw new Error('useConversationContext must be used inside <ConversationProvider>.')
  return conversation
}
