import type { ReactNode } from 'react'

import type { AgentApi } from '../api/agentApi'
import { ConversationContext } from './context'
import { useConversation } from './useConversation'

export function ConversationProvider({ children, api }: { children: ReactNode; api?: AgentApi }) {
  const conversation = useConversation(api)
  return <ConversationContext.Provider value={conversation}>{children}</ConversationContext.Provider>
}
