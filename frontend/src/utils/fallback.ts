/**
 * Prefix FallbackBackend (app/agent.py) adds when Ollama failed during a turn
 * and the rule-based backend answered instead. The API has no separate field
 * for this, so the reply text is the only signal.
 */
export const FALLBACK_PREFIX = '[LLM unavailable:'

export function isFallbackReply(text: string): boolean {
  return text.startsWith(FALLBACK_PREFIX)
}
