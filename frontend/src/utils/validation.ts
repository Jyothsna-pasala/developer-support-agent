/** Same limit as MAX_INPUT_CHARS in app/agent.py. */
export const MAX_MESSAGE_CHARS = 2000

/**
 * Count characters the way the backend does. Python counts code points, so
 * an emoji is 1 there but 2 in JavaScript's `.length`.
 */
export function countChars(text: string): number {
  return [...text].length
}

/**
 * Validate a message with the backend's rules (trimmed, 1-2000 characters).
 * Returns the backend's own error text, or null when valid.
 */
export function validateMessage(text: string): string | null {
  const trimmed = text.trim()
  if (!trimmed) return 'Please enter a question.'
  if (countChars(trimmed) > MAX_MESSAGE_CHARS) {
    return `Please keep questions under ${MAX_MESSAGE_CHARS} characters.`
  }
  return null
}
