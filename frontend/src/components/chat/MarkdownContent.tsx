import 'katex/dist/katex.min.css'

import Box from '@mui/material/Box'
import type { Components, Options } from 'react-markdown'
import Markdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { MONOSPACE } from '../../theme'
import { remarkDollarGuard } from '../../utils/remarkDollarGuard'

// Math: $…$ inline, $$…$$ (or a ```math block) for display. Code spans and
// other code blocks stay code, even if they contain $ or LaTeX.
const remarkPlugins: Options['remarkPlugins'] = [remarkGfm, remarkMath, remarkDollarGuard]
// KaTeX's `trust` stays off, so \href, \includegraphics and similar commands
// are not rendered. Invalid LaTeX is shown as red source text instead of throwing.
const rehypePlugins: Options['rehypePlugins'] = [[rehypeKatex, { strict: 'ignore' }]]

// Raw HTML in replies is not rendered (react-markdown's default), so model
// output cannot inject markup.
const components: Components = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
  // Wide tables scroll inside the bubble instead of stretching the page.
  table: ({ node: _node, ...props }) => (
    <Box className="md-table-wrap">
      <table {...props} />
    </Box>
  ),
}

/** Renders an agent reply as GitHub-flavoured Markdown, styled to match MUI. */
export function MarkdownContent({ text }: { text: string }) {
  return (
    <Box
      className="markdown"
      sx={{
        fontSize: '0.95rem',
        lineHeight: 1.6,
        overflowWrap: 'anywhere',
        '& > :first-of-type': { mt: 0 },
        '& > :last-child': { mb: 0 },
        '& p': { my: 1 },
        '& h1, & h2, & h3, & h4, & h5, & h6': { mt: 2, mb: 1, lineHeight: 1.3, fontWeight: 600 },
        '& h1': { fontSize: '1.35rem' },
        '& h2': { fontSize: '1.2rem' },
        '& h3': { fontSize: '1.08rem' },
        '& h4, & h5, & h6': { fontSize: '1rem' },
        '& ul, & ol': { my: 1, pl: 3 },
        '& li + li': { mt: 0.5 },
        '& a': { color: 'primary.main' },
        '& blockquote': {
          my: 1,
          mx: 0,
          pl: 1.5,
          borderLeft: 3,
          borderColor: 'divider',
          color: 'text.secondary',
        },
        '& code': {
          fontFamily: MONOSPACE,
          fontSize: '0.85em',
          bgcolor: 'action.hover',
          px: 0.5,
          py: 0.1,
          borderRadius: 0.5,
        },
        '& pre': {
          my: 1.5,
          p: 1.5,
          overflowX: 'auto',
          bgcolor: 'action.hover',
          border: 1,
          borderColor: 'divider',
          borderRadius: 1,
        },
        '& pre code': { bgcolor: 'transparent', p: 0, fontSize: '0.85rem', whiteSpace: 'pre' },
        '& .md-table-wrap': { my: 1.5, overflowX: 'auto' },
        '& table': { borderCollapse: 'collapse', minWidth: '50%', fontSize: '0.9rem' },
        '& th, & td': { border: 1, borderColor: 'divider', px: 1, py: 0.5, textAlign: 'left', verticalAlign: 'top' },
        '& th': { bgcolor: 'action.hover', fontWeight: 600 },
        '& hr': { border: 0, borderTop: 1, borderColor: 'divider', my: 2 },
        // Wide formulas scroll inside the bubble instead of stretching the page.
        '& .katex-display': { my: 1.5, py: 0.5, overflowX: 'auto', overflowY: 'hidden' },
        '& .katex': { fontSize: '1.05em' },
      }}
    >
      <Markdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>
        {text}
      </Markdown>
    </Box>
  )
}
