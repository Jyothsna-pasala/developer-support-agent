import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // The API only allows the origins in API_CORS_ORIGINS (port 5173 by
  // default), so never silently move to another port.
  server: { port: 5173, strictPort: true },
  preview: { port: 5173, strictPort: true },
  build: {
    rolldownOptions: {
      output: {
        // Separate long-lived vendor chunks so app changes don't re-download them.
        codeSplitting: {
          groups: [
            { name: 'mui', test: /node_modules[\\/](@mui|@emotion)[\\/]/ },
            { name: 'katex', test: /node_modules[\\/]katex[\\/]/ },
            { name: 'markdown', test: /node_modules[\\/](react-markdown|remark-|rehype-|mdast-|micromark|unified|hast-|unist-|vfile)/ },
            { name: 'react', test: /node_modules[\\/](react|react-dom|react-router|scheduler)[\\/]/ },
          ],
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    restoreMocks: true,
  },
})
