import { describe, expect, it } from 'vitest'

import { isPlausibleInlineMath } from './remarkDollarGuard'

describe('isPlausibleInlineMath', () => {
  it('accepts single-dollar math, with or without inner spaces', () => {
    expect(isPlausibleInlineMath('$x$', ' ')).toBe(true)
    expect(isPlausibleInlineMath('$E = mc^2$', '.')).toBe(true)
    expect(isPlausibleInlineMath('$2x + 1$', '')).toBe(true)
    // qwen3:4b's style, from a real reply.
    expect(isPlausibleInlineMath('$ ax^2 + bx + c = 0 $', ' ')).toBe(true)
    expect(isPlausibleInlineMath('$ 2a $', ')')).toBe(true)
  })

  it('rejects money-like pairs', () => {
    expect(isPlausibleInlineMath('$5 and $', '1')).toBe(false) // "$5 and $10"
    expect(isPlausibleInlineMath('$1,000 and $', '2')).toBe(false) // "$1,000 and $2,000"
    expect(isPlausibleInlineMath('$5 or $', ' ')).toBe(false) // "$5 or $ 10"
    expect(isPlausibleInlineMath('$ $', ' ')).toBe(false)
  })

  it('leaves double-dollar math alone', () => {
    expect(isPlausibleInlineMath('$$ x $$', '1')).toBe(true)
  })
})
