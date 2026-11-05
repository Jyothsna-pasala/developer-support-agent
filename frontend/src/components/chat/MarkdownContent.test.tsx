import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { MarkdownContent } from './MarkdownContent'

const REPLY = [
  '## Fixing CORS',
  '',
  'Use the `Access-Control-Allow-Origin` header. **Server-side** only.',
  '',
  '1. Add the header',
  '2. Handle preflight `OPTIONS`',
  '',
  '- bullet one',
  '- bullet two',
  '',
  '```python',
  'from fastapi.middleware.cors import CORSMiddleware',
  'app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"])',
  '```',
  '',
  '| Status | Meaning |',
  '|---|---|',
  '| 404 | Not found |',
  '| 409 | Conflict |',
  '',
  'See [MDN](https://developer.mozilla.org/).',
].join('\n')

describe('MarkdownContent', () => {
  it('renders headings, lists, inline code, code blocks, tables and links', () => {
    const { container } = render(<MarkdownContent text={REPLY} />)

    expect(screen.getByRole('heading', { level: 2, name: 'Fixing CORS' })).toBeInTheDocument()
    expect(screen.getByText('Access-Control-Allow-Origin').tagName).toBe('CODE')
    expect(screen.getByText('Server-side').tagName).toBe('STRONG')

    const lists = screen.getAllByRole('list')
    expect(lists.map((l) => l.tagName)).toEqual(['OL', 'UL'])
    expect(within(lists[0]).getAllByRole('listitem')).toHaveLength(2)

    const pre = container.querySelector('pre code')
    expect(pre?.textContent).toContain('CORSMiddleware')
    expect(pre?.className).toContain('language-python')

    const table = screen.getByRole('table')
    expect(within(table).getAllByRole('columnheader').map((c) => c.textContent)).toEqual(['Status', 'Meaning'])
    expect(within(table).getAllByRole('row')).toHaveLength(3)

    const link = screen.getByRole('link', { name: 'MDN' })
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('does not render raw HTML from the model', () => {
    const { container } = render(<MarkdownContent text={'<img src=x onerror="alert(1)"> hi'} />)
    expect(container.querySelector('img')).toBeNull()
  })

  it('renders the rule-based backend plain text', () => {
    render(<MarkdownContent text={'(1200 * 0.18) + 50 = 266'} />)
    expect(screen.getByText('(1200 * 0.18) + 50 = 266')).toBeInTheDocument()
  })
})

describe('MarkdownContent: math', () => {
  const annotation = (root: Element) => root.querySelector('annotation[encoding="application/x-tex"]')?.textContent

  it('renders inline $…$ math inside the sentence', () => {
    const { container } = render(<MarkdownContent text={'Energy is $E = mc^2$ here.'} />)
    const math = container.querySelector('p .katex')
    expect(math).not.toBeNull()
    expect(container.querySelector('.katex-display')).toBeNull()
    expect(annotation(math!)).toBe('E = mc^2')
    expect(container.querySelector('p')).toHaveTextContent(/^Energy is .* here\.$/)
  })

  it('renders $$…$$ and ```math blocks as display math', () => {
    const quadratic = String.raw`x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}`
    const text = ['Formula:', '', '$$', quadratic, '$$', '', '```math', 'a^2 + b^2 = c^2', '```'].join('\n')
    const { container } = render(<MarkdownContent text={text} />)
    const blocks = container.querySelectorAll('.katex-display')
    expect(blocks).toHaveLength(2)
    expect(annotation(blocks[0])).toBe(quadratic)
    expect(annotation(blocks[1])?.trim()).toBe('a^2 + b^2 = c^2')
    expect(container.querySelector('pre')).toBeNull()
  })

  it('keeps math syntax inside code spans and other code blocks as code', () => {
    const python = 'price = f"${total}"  # $a$ and ' + String.raw`\frac{1}{2}`
    const latex = String.raw`$$\int_0^1 x\,dx$$`
    const text = ['Use `$x^2$` literally.', '', '```python', python, '```', '', '```latex', latex, '```'].join('\n')
    const { container } = render(<MarkdownContent text={text} />)
    expect(container.querySelector('.katex')).toBeNull()
    expect(screen.getByText('$x^2$').tagName).toBe('CODE')
    const blocks = [...container.querySelectorAll('pre code')]
    expect(blocks.map((b) => b.className)).toEqual(['language-python', 'language-latex'])
    expect(blocks[0].textContent).toBe(`${python}\n`)
    expect(blocks[1].textContent).toBe(`${latex}\n`)
  })

  it('leaves dollar amounts as text', () => {
    const text = 'It costs $5 and $10 today, between $1,000 and $2,000, or $5 or $ 10. Total: $266.'
    const { container } = render(<MarkdownContent text={text} />)
    expect(container.querySelector('.katex')).toBeNull()
    expect(container.querySelector('p')).toHaveTextContent(text)
  })

  it('renders a real qwen3:4b reply with spaced $ … $ inline math', () => {
    // Excerpt of an actual reply from the ollama backend (GET /history).
    const reply = [
      'The quadratic formula is:',
      '',
      '**x = [-b ± √(b² - 4ac)] / (2a)**',
      '',
      '### Brief explanation of terms:',
      String.raw`- **a, b, c**: Coefficients of the quadratic equation $ ax^2 + bx + c = 0 $ (where $ a \neq 0 $).`,
      '- **Discriminant ($ b^2 - 4ac $)**: Determines the nature of the roots:',
      '  - Positive → Two distinct real roots',
      '- **Denominator ($ 2a $)**: Scales the result based on the coefficient of $ x^2 $.',
    ].join('\n')
    const { container } = render(<MarkdownContent text={reply} />)
    const tex = [...container.querySelectorAll('annotation')].map((a) => a.textContent?.trim())
    expect(tex).toEqual(['ax^2 + bx + c = 0', String.raw`a \neq 0`, 'b^2 - 4ac', '2a', 'x^2'])
    expect(container.querySelector('.katex-error')).toBeNull()
    expect(container.textContent).not.toContain('$')
    expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent('Brief explanation of terms:')
  })

  it('shows invalid LaTeX as an error instead of crashing', () => {
    const { container } = render(<MarkdownContent text={String.raw`Broken $\frac{1$ formula`} />)
    expect(container.querySelector('.katex-error')).not.toBeNull()
    expect(container.querySelector('p')).toHaveTextContent('formula')
  })

  it('does not render links or HTML from LaTeX commands (KaTeX trust is off)', () => {
    const { container } = render(<MarkdownContent text={String.raw`$\href{javascript:alert(1)}{click}$`} />)
    expect(container.querySelector('.katex')).not.toBeNull()
    expect(container.querySelector('a')).toBeNull()
  })
})
