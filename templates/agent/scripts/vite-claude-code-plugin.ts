import type { IncomingMessage, ServerResponse } from 'http'
import type { Plugin, ViteDevServer } from 'vite'

/**
 * Vite plugin that intercepts /stream requests for the claude-code provider.
 * Runs in Node.js (Vite's dev server), not in the Cloudflare worker,
 * since the claude-code provider needs child_process to spawn the Claude CLI.
 */
export function claudeCodePlugin(): Plugin {
	return {
		name: 'claude-code-stream',
		configureServer(server: ViteDevServer) {
			server.middlewares.use(
				async (req: IncomingMessage, res: ServerResponse, next: () => void) => {
					// Only intercept POST /stream with X-Provider: claude-code header
					if (
						req.method !== 'POST' ||
						req.url !== '/stream' ||
						req.headers['x-provider'] !== 'claude-code'
					) {
						return next()
					}

					try {
						console.log('[claude-code] Intercepted /stream request')

						// Read the request body
						const chunks: Buffer[] = []
						for await (const chunk of req) {
							chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk)
						}
						const prompt = JSON.parse(Buffer.concat(chunks).toString())
						console.log('[claude-code] Parsed prompt, loading modules...')

						// Use ssrLoadModule to import TS modules with Vite's transpilation
						const { buildSystemPrompt } = (await server.ssrLoadModule(
							'/worker/prompt/buildSystemPrompt.ts'
						)) as typeof import('../worker/prompt/buildSystemPrompt')
						const { buildMessages } = (await server.ssrLoadModule(
							'/worker/prompt/buildMessages.ts'
						)) as typeof import('../worker/prompt/buildMessages')

						// Dynamic import for Node.js-only modules
						const { claudeCode } = await import('ai-sdk-provider-claude-code')
						const { streamText } = await import('ai')
						console.log('[claude-code] Modules loaded, creating model...')

						const systemPrompt = buildSystemPrompt(prompt)

						const model = claudeCode('sonnet', {
							allowDangerouslySkipPermissions: true,
							permissionMode: 'bypassPermissions',
						})

						const messages: any[] = []
						const promptMessages = buildMessages(prompt)
						messages.push(...promptMessages)

						console.log(
							'[claude-code] Calling streamText with',
							messages.length,
							'messages, system prompt length:',
							systemPrompt.length
						)

						const { textStream } = streamText({
							model,
							system: systemPrompt,
							messages,
							maxOutputTokens: 8192,
						})

						// Set SSE headers
						res.writeHead(200, {
							'Content-Type': 'text/event-stream',
							'Cache-Control': 'no-cache, no-transform',
							Connection: 'keep-alive',
							'Transfer-Encoding': 'chunked',
							'Access-Control-Allow-Origin': '*',
						})

						// Claude Code responses mix plain text with JSON blocks.
						// We need to extract JSON `{"actions": [...]}` blocks from the stream
						// and parse actions from them, ignoring surrounding prose.
						let buffer = ''
						let startTime = Date.now()
						let chunkCount = 0
						let actionsSent = 0

						for await (const text of textStream) {
							chunkCount++
							buffer += text
							if (chunkCount <= 3) {
								console.log(`[claude-code] Chunk ${chunkCount}: ${text.slice(0, 200)}`)
							}

							// Try to find and parse complete JSON blocks from the buffer
							const extracted = extractJsonActions(buffer)
							for (const action of extracted.actions) {
								actionsSent++
								const event = { ...action, complete: true, time: Date.now() - startTime }
								res.write(`data: ${JSON.stringify(event)}\n\n`)
							}
							// Keep only the unparsed remainder
							buffer = extracted.remainder
						}

						console.log(
							`[claude-code] Stream ended. Total chunks: ${chunkCount}, actions sent: ${actionsSent}`
						)
						if (buffer.length > 0) {
							console.log('[claude-code] Remaining buffer (first 500 chars):', buffer.slice(0, 500))
						}

						// Fallback: if no actions were extracted, wrap the full text as a message
						if (actionsSent === 0 && buffer.trim().length > 0) {
							// Strip any JSON artifacts from the text for clean display
							const cleanText = buffer
								.replace(/```json\s*/g, '')
								.replace(/```\s*/g, '')
								.trim()
							if (cleanText.length > 0) {
								console.log('[claude-code] No actions parsed, sending as message')
								const messageAction = {
									_type: 'message',
									text: cleanText,
									complete: true,
									time: Date.now() - startTime,
								}
								res.write(`data: ${JSON.stringify(messageAction)}\n\n`)
							}
						}

						res.end()
					} catch (error: any) {
						console.error('[claude-code] Stream error:', error)
						if (!res.headersSent) {
							res.writeHead(500, { 'Content-Type': 'application/json' })
						}
						res.end(JSON.stringify({ error: error.message }))
					}
				}
			)
		},
	}
}

/**
 * Extract complete JSON `{"actions": [...]}` blocks from a text buffer
 * that may contain a mix of prose and JSON.
 * Returns the parsed actions and the remaining unparsed text.
 */
function extractJsonActions(buffer: string): { actions: any[]; remainder: string } {
	const actions: any[] = []
	let remainder = buffer

	// Look for JSON objects that start with {"actions"
	// They may appear after prose text, possibly inside ```json code blocks
	const jsonPattern = /\{[\s\n]*"actions"\s*:\s*\[/g
	let match

	while ((match = jsonPattern.exec(remainder)) !== null) {
		const startIdx = match.index
		// Try to find the end of this JSON object by counting braces
		let depth = 0
		let inString = false
		let endIdx = -1

		for (let i = startIdx; i < remainder.length; i++) {
			const char = remainder[i]

			if (char === '"' && (i === 0 || remainder[i - 1] !== '\\')) {
				inString = !inString
				continue
			}
			if (inString) continue

			if (char === '{') depth++
			else if (char === '}') {
				depth--
				if (depth === 0) {
					endIdx = i + 1
					break
				}
			}
		}

		if (endIdx === -1) {
			// Incomplete JSON block - stop here, keep remainder for next iteration
			break
		}

		const jsonStr = remainder.slice(startIdx, endIdx)
		try {
			const parsed = JSON.parse(jsonStr)
			if (parsed.actions && Array.isArray(parsed.actions)) {
				actions.push(...parsed.actions)
			}
		} catch {
			// Malformed JSON, skip it
			console.log('[claude-code] Failed to parse JSON block:', jsonStr.slice(0, 100))
		}

		// Remove everything up to and including this JSON block
		remainder = remainder.slice(endIdx)
		jsonPattern.lastIndex = 0 // Reset regex since we modified the string
	}

	return { actions, remainder }
}
