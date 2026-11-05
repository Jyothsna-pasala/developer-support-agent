import BuildIcon from '@mui/icons-material/Build'
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlined'
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutlined'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useId, useState, type ReactNode } from 'react'

import type { ToolCall } from '../../api/types'
import type { ToolSummary } from '../../conversation/types'
import { MONOSPACE } from '../../theme'
import { JsonBlock, ToolResultView } from './ToolResultView'

const RESTORED_RESULT_TEXT =
  'Not available after a refresh. The conversation history only records the tool name, arguments and status.'

export function ToolStatusChip({ ok }: { ok: boolean }) {
  return (
    <Chip
      size="small"
      variant="outlined"
      color={ok ? 'success' : 'error'}
      icon={ok ? <CheckCircleOutlineIcon /> : <ErrorOutlineIcon />}
      label={ok ? 'Success' : 'Failed'}
      sx={{ ml: 'auto', flexShrink: 0 }}
    />
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <Typography variant="overline" color="text.secondary" component="div">
        {title}
      </Typography>
      {children}
    </div>
  )
}

/** Collapsed: tool name, one-line arguments and status. Expanded: `children`. */
function ToolCardFrame({
  name,
  ok,
  argumentsPreview,
  children,
}: {
  name: string
  ok: boolean
  argumentsPreview: string
  children: ReactNode
}) {
  const id = useId()
  return (
    <Accordion
      disableGutters
      variant="outlined"
      sx={{ '&:before': { display: 'none' }, bgcolor: 'background.default', minWidth: 0 }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls={`${id}-details`}
        id={`${id}-summary`}
        // Let the one-line arguments shrink with an ellipsis on narrow screens.
        sx={{ '& .MuiAccordionSummary-content': { minWidth: 0 } }}
      >
        <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', minWidth: 0, width: '100%' }}>
          <BuildIcon fontSize="small" color="action" />
          <Typography sx={{ fontFamily: MONOSPACE, fontSize: '0.85rem', fontWeight: 600, minWidth: 0 }} noWrap>
            {name}
          </Typography>
          <Typography
            sx={{
              fontFamily: MONOSPACE,
              fontSize: '0.8rem',
              color: 'text.secondary',
              minWidth: 0,
              flex: 1,
              // Phones show only the name and status; the arguments are one tap away.
              display: { xs: 'none', sm: 'block' },
            }}
            noWrap
            title={argumentsPreview}
          >
            {argumentsPreview}
          </Typography>
          <ToolStatusChip ok={ok} />
        </Stack>
      </AccordionSummary>
      <AccordionDetails id={`${id}-details`} sx={{ pt: 0 }}>
        <Stack spacing={1.5}>{children}</Stack>
      </AccordionDetails>
    </Accordion>
  )
}

/** One tool execution from ChatResponse.tool_calls: name, arguments and result. */
export function ToolCallCard({ call }: { call: ToolCall }) {
  const [showRaw, setShowRaw] = useState(false)

  return (
    <ToolCardFrame name={call.name} ok={call.ok} argumentsPreview={JSON.stringify(call.arguments)}>
      <Section title="Arguments">
        <JsonBlock value={call.arguments} label="Tool arguments" />
      </Section>
      <Section title={call.output.ok ? 'Result' : 'Error'}>
        <ToolResultView call={call} />
      </Section>
      <div>
        <Button size="small" onClick={() => setShowRaw((v) => !v)}>
          {showRaw ? 'Hide raw output' : 'Show raw output'}
        </Button>
        {showRaw && <JsonBlock value={call.output} label="Raw tool output" />}
      </div>
    </ToolCardFrame>
  )
}

function parseArguments(text: string): { ok: true; value: unknown } | { ok: false } {
  try {
    return { ok: true, value: JSON.parse(text) }
  } catch {
    return { ok: false }
  }
}

/** A tool entry restored from GET /history: only name, arguments and status exist. */
export function RestoredToolCallCard({ summary }: { summary: ToolSummary }) {
  const parsed = parseArguments(summary.argumentsText)
  return (
    <ToolCardFrame name={summary.name} ok={summary.ok} argumentsPreview={summary.argumentsText}>
      <Section title="Arguments">
        {parsed.ok ? (
          <JsonBlock value={parsed.value} label="Tool arguments" />
        ) : (
          <Box
            component="pre"
            aria-label="Tool arguments"
            sx={{ m: 0, p: 1, fontFamily: MONOSPACE, fontSize: '0.8rem', bgcolor: 'action.hover', borderRadius: 1, overflowX: 'auto' }}
          >
            {summary.argumentsText}
          </Box>
        )}
      </Section>
      <Section title={summary.ok ? 'Result' : 'Error'}>
        <Typography variant="body2" color="text.secondary">
          {RESTORED_RESULT_TEXT}
        </Typography>
      </Section>
    </ToolCardFrame>
  )
}
