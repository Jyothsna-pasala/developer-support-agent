import Box from '@mui/material/Box'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import Chip from '@mui/material/Chip'
import Container from '@mui/material/Container'
import Grid from '@mui/material/Grid'
import Skeleton from '@mui/material/Skeleton'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'

import type { ToolInfo } from '../api/types'
import { ErrorAlert } from '../components/feedback/ErrorAlert'
import { useTools } from '../hooks/useTools'
import { MONOSPACE } from '../theme'

interface SchemaProperty {
  name: string
  type?: string
  description?: string
  required: boolean
}

/** Read input properties from the tool's JSON Schema, tolerating other shapes. */
function schemaProperties(parameters: Record<string, unknown>): SchemaProperty[] {
  const properties = parameters.properties
  if (typeof properties !== 'object' || properties === null) return []
  const required = Array.isArray(parameters.required) ? parameters.required : []
  return Object.entries(properties as Record<string, Record<string, unknown>>).map(([name, schema]) => ({
    name,
    type: typeof schema?.type === 'string' ? schema.type : undefined,
    description: typeof schema?.description === 'string' ? schema.description : undefined,
    required: required.includes(name),
  }))
}

function ToolCard({ tool }: { tool: ToolInfo }) {
  const inputs = schemaProperties(tool.parameters)
  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" component="h3" sx={{ fontFamily: MONOSPACE, fontSize: '1rem', mb: 1 }}>
          {tool.name}
        </Typography>
        <Typography variant="body2" sx={{ mb: 2 }}>
          {tool.description}
        </Typography>
        <Typography variant="overline" color="text.secondary">
          Input
        </Typography>
        {inputs.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No arguments
          </Typography>
        ) : (
          <Stack spacing={0.75}>
            {inputs.map((input) => (
              <Box key={input.name}>
                <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                  <Typography sx={{ fontFamily: MONOSPACE, fontSize: '0.85rem' }}>{input.name}</Typography>
                  {input.type && <Chip size="small" label={input.type} variant="outlined" />}
                  {input.required && <Chip size="small" label="required" />}
                </Stack>
                {input.description && (
                  <Typography variant="body2" color="text.secondary">
                    {input.description}
                  </Typography>
                )}
              </Box>
            ))}
          </Stack>
        )}
        <Typography variant="overline" color="text.secondary" component="div" sx={{ mt: 1.5 }}>
          Output
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {tool.output_description}
        </Typography>
      </CardContent>
    </Card>
  )
}

export function ToolsPage() {
  const { tools, error, loading, reload } = useTools()

  return (
    <Container maxWidth="lg" sx={{ py: 3, px: { xs: 1.5, sm: 3 } }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Agent tools
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 3 }}>
        The model decides when to call these. Each chat reply shows the calls it made, with arguments and results.
      </Typography>

      {error && <ErrorAlert error={error} onRetry={reload} />}

      {loading && !tools && (
        <Grid container spacing={2} aria-label="Loading tools">
          {[0, 1, 2].map((i) => (
            <Grid key={i} size={{ xs: 12, sm: 6, lg: 4 }}>
              <Skeleton variant="rounded" height={220} />
            </Grid>
          ))}
        </Grid>
      )}

      {tools && tools.length === 0 && <Typography color="text.secondary">The API reports no tools.</Typography>}

      {tools && tools.length > 0 && (
        <Grid container spacing={2}>
          {tools.map((tool) => (
            <Grid key={tool.name} size={{ xs: 12, sm: 6, lg: 4 }}>
              <ToolCard tool={tool} />
            </Grid>
          ))}
        </Grid>
      )}
    </Container>
  )
}
