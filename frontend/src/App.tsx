import { Route, Routes } from 'react-router-dom'

import { AppLayout } from './components/layout/AppLayout'
import { ConversationProvider } from './conversation/ConversationProvider'
import { ChatPage } from './pages/ChatPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { ToolsPage } from './pages/ToolsPage'

export default function App() {
  return (
    // Above the routes, so switching to Tools never drops a running turn.
    <ConversationProvider>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<ChatPage />} />
          <Route path="tools" element={<ToolsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </ConversationProvider>
  )
}
