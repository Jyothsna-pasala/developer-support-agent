import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'

import type { ToolCall } from '../../api/types'
import { MONOSPACE } from '../../theme'
import {
  formatJson,
  isCalculationResult,
  isDateTimeResult,
  isKnowledgeBaseResult,
} from '../../utils/toolResults'

export function JsonBlock({ value, label }: { value: unknown; label?: string }) {
  return (
    <Box
      component="pre"
      aria-label={label}
      sx={{
        m: 0,
        p: 1,
        fontFamily: MONOSPACE,
        fontSize: '0.8rem',
        bgcolor: 'action.hover',
        borderRadius: 1,
        overflowX: 'auto',
        whiteSpace: 'pre',
      }}
    >
      {formatJson(value)}
    </Box>
  )
}

/** Dedicated view for the tool's result, or raw JSON for unknown tools/shapes. */
export function ToolResultView({ call }: { call: ToolCall }) {
  const output = call.output
  if (!output.ok) {
    return (
      <Alert severity="error" variant="outlined" sx={{ py: 0 }}>
        {output.error}
      </Alert>
    )
  }

  const result = output.result
  if (call.name === 'calculate_expression' && isCalculationResult(result)) {
    return (
      <Typography sx={{ fontFamily: MONOSPACE, fontSize: '1rem' }}>
        {result.expression} = <strong>{result.result}</strong>
      </Typography>
    )
  }

  if (call.name === 'get_current_datetime' && isDateTimeResult(result)) {
    return (
      <Stack spacing={0.25}>
        <Typography>
          {result.weekday}, {result.date} at {result.time}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {result.timezone} (UTC{result.utc_offset}) · {result.iso_8601}
        </Typography>
      </Stack>
    )
  }

  if (call.name === 'search_knowledge_base' && isKnowledgeBaseResult(result)) {
    return (
      <Stack spacing={1}>
        <Typography variant="body2" color="text.secondary">
          {result.match_count} match{result.match_count === 1 ? '' : 'es'} for “{result.query}”
        </Typography>
        {result.results.map((article) => (
          <Box key={article.title} sx={{ borderLeft: 3, borderColor: 'primary.light', pl: 1.25 }}>
            <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap', rowGap: 0.5 }}>
              <Typography variant="subtitle2">{article.title}</Typography>
              <Chip size="small" label={article.category} variant="outlined" />
              <Typography variant="caption" color="text.secondary">
                score {article.score}
              </Typography>
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.25 }}>
              {article.content}
            </Typography>
          </Box>
        ))}
      </Stack>
    )
  }

  return <JsonBlock value={result} label="Tool result" />
}
