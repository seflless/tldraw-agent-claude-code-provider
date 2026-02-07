import type { IncomingMessage, ServerResponse } from 'http'
import path from 'path'
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
							cwd: path.resolve(process.cwd(), '../../'),
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

						// Use fullStream instead of textStream to get tool activity events too.
						// textStream only yields text deltas — during tool execution phases
						// (which can be long), nothing comes through and the UI appears frozen.
						const result = streamText({
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

						const send = (data: any) => {
							res.write(`data: ${JSON.stringify(data)}\n\n`)
						}

						let buffer = ''
						let startTime = Date.now()
						let actionsSent = 0
						let lastPartialAction: any = null

						for await (const event of result.fullStream) {
							// --- Text deltas: accumulate and extract actions ---
							if (event.type === 'text-delta') {
								buffer += event.text

								// Extract any complete individual actions from the buffer
								const extracted = extractJsonActions(buffer)
								for (const action of extracted.completeActions) {
									console.log('[claude-code] Action:', action._type)
									actionsSent++
									lastPartialAction = null
									startTime = Date.now()
									send({ ...action, complete: true, time: Date.now() - startTime })
								}

								// Preview the in-progress action (incomplete JSON closed heuristically)
								if (extracted.partialAction) {
									lastPartialAction = extracted.partialAction
									send({
										...extracted.partialAction,
										complete: false,
										time: Date.now() - startTime,
									})
								}

								buffer = extracted.remainder
								continue
							}

							// --- Tool activity: show what Claude Code is doing ---
							if (event.type === 'tool-call') {
								const toolName = event.toolName ?? 'unknown'
								console.log(`[claude-code] Tool call: ${toolName}`)
								send({
									_type: 'message',
									text: `🔧 ${toolName}`,
									complete: true,
									time: Date.now() - startTime,
								})
								continue
							}
						}

						console.log(`[claude-code] Stream ended. Actions sent: ${actionsSent}`)

						// Finalize the last partial action if the stream ended mid-action
						if (lastPartialAction) {
							send({ ...lastPartialAction, complete: true, time: Date.now() - startTime })
							actionsSent++
						}

						// Fallback: if no actions were extracted, wrap the full text as a message
						if (actionsSent === 0 && buffer.trim().length > 0) {
							const cleanText = buffer
								.replace(/```json\s*/g, '')
								.replace(/```\s*/g, '')
								.trim()
							if (cleanText.length > 0) {
								console.log('[claude-code] No actions parsed, sending as message')
								send({
									_type: 'message',
									text: cleanText,
									complete: true,
									time: Date.now() - startTime,
								})
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
 * Extract individual complete action objects from an `{"actions": [...]}` block
 * incrementally, plus a partial preview of the in-progress action.
 */
function extractJsonActions(buffer: string): {
	completeActions: any[]
	partialAction: any | null
	remainder: string
} {
	const completeActions: any[] = []

	const arrayMatch = /\{[\s\n]*"actions"\s*:\s*\[/.exec(buffer)
	if (!arrayMatch) {
		return { completeActions, partialAction: null, remainder: buffer }
	}

	let cursor = arrayMatch.index + arrayMatch[0].length
	let arrayEnded = false

	while (cursor < buffer.length) {
		// Skip whitespace and commas between array elements
		while (cursor < buffer.length && /[\s,]/.test(buffer[cursor])) {
			cursor++
		}
		if (cursor >= buffer.length) break

		if (buffer[cursor] === ']') {
			arrayEnded = true
			cursor++
			while (cursor < buffer.length && /\s/.test(buffer[cursor])) cursor++
			if (cursor < buffer.length && buffer[cursor] === '}') cursor++
			break
		}

		if (buffer[cursor] !== '{') {
			cursor++
			continue
		}

		// Count braces to find the end of this action object
		let depth = 0
		let inString = false
		let endIdx = -1

		for (let i = cursor; i < buffer.length; i++) {
			const char = buffer[i]
			if (char === '"' && (i === 0 || buffer[i - 1] !== '\\')) {
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

		if (endIdx === -1) break // incomplete object

		try {
			completeActions.push(JSON.parse(buffer.slice(cursor, endIdx)))
		} catch {
			// skip malformed
		}
		cursor = endIdx
	}

	// Try to parse the in-progress (incomplete) action via heuristic closing
	let partialAction: any | null = null
	if (!arrayEnded && cursor < buffer.length && buffer[cursor] === '{') {
		partialAction = closeAndParseJson(buffer.slice(cursor))
	}

	if (arrayEnded) {
		const afterBlock = buffer.slice(cursor)
		if (afterBlock.includes('"actions"')) {
			const more = extractJsonActions(afterBlock)
			completeActions.push(...more.completeActions)
			return { completeActions, partialAction: more.partialAction, remainder: more.remainder }
		}
		return { completeActions, partialAction: null, remainder: afterBlock }
	}

	return { completeActions, partialAction, remainder: '{"actions": [' + buffer.slice(cursor) }
}

/**
 * Parse potentially incomplete JSON by closing all unclosed brackets/braces/strings.
 */
function closeAndParseJson(str: string): any | null {
	const stack: string[] = []
	let i = 0
	while (i < str.length) {
		const char = str[i]
		const last = stack.at(-1)
		if (char === '"') {
			if (i > 0 && str[i - 1] === '\\') {
				i++
				continue
			}
			if (last === '"') stack.pop()
			else stack.push('"')
		}
		if (last === '"') {
			i++
			continue
		}
		if (char === '{' || char === '[') stack.push(char)
		if (char === '}' && last === '{') stack.pop()
		if (char === ']' && last === '[') stack.pop()
		i++
	}
	let closed = str
	for (let j = stack.length - 1; j >= 0; j--) {
		const o = stack[j]
		if (o === '{') closed += '}'
		else if (o === '[') closed += ']'
		else if (o === '"') closed += '"'
	}
	try {
		return JSON.parse(closed)
	} catch {
		return null
	}
}
