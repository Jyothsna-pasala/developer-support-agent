/**
 * remark-math treats any two single `$` in a paragraph as inline math, so
 * "It costs $5 and $10" would render "5 and " as a formula. This plugin runs
 * after parsing and turns single-dollar math that looks like money back into
 * plain text.
 *
 * Pandoc's stricter rule (no space inside the delimiters) is not used because
 * qwen3 writes inline math as `$ ax^2 + bx + c = 0 $`.
 *
 * `$$…$$` and block math are left alone, and code nodes are never touched
 * (remark-math never creates math inside code).
 */

// Minimal structural types for the mdast nodes this plugin touches.
interface Point {
  offset?: number
}
interface MdNode {
  type: string
  value?: string
  children?: MdNode[]
  position?: { start: Point; end: Point }
}

/**
 * True unless a single-dollar inline math node looks like two amounts of
 * money: empty, a digit right after the closing `$` ("$5 and $10"), or content
 * that starts with a digit and ends with a space ("$5 or $ 10").
 */
export function isPlausibleInlineMath(raw: string, nextChar: string): boolean {
  if (!raw.startsWith('$') || raw.startsWith('$$')) return true
  const inner = raw.slice(1, -1)
  if (inner.trim().length === 0) return false
  if (/[0-9]/.test(nextChar)) return false
  return !(/^[0-9]/.test(inner) && /\s$/.test(inner))
}

function revertImplausibleMath(node: MdNode, source: string): void {
  if (!node.children) return
  node.children = node.children.map((child) => {
    const start = child.position?.start.offset
    const end = child.position?.end.offset
    if (child.type === 'inlineMath' && start !== undefined && end !== undefined) {
      const raw = source.slice(start, end)
      if (!isPlausibleInlineMath(raw, source.charAt(end))) {
        return { type: 'text', value: raw, position: child.position }
      }
      return child
    }
    revertImplausibleMath(child, source)
    return child
  })
}

/** Use after remark-math in `remarkPlugins`. */
export function remarkDollarGuard() {
  return (tree: MdNode, file: { value: unknown }) => {
    revertImplausibleMath(tree, String(file.value))
  }
}
