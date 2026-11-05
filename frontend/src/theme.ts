import { createTheme } from '@mui/material/styles'

// Light and dark palettes; MUI follows the operating system preference.
export const theme = createTheme({
  cssVariables: true,
  colorSchemes: { light: true, dark: true },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  components: {
    MuiButton: { defaultProps: { disableElevation: true } },
  },
})

export const MONOSPACE = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'
