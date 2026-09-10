<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { api } from '../api'

const sessions = ref<any[]>([])
const activeSession = ref<number | null>(null)
const messages = ref<any[]>([])
const question = ref('')
const sending = ref(false)
const controlling = ref<number | null>(null)
const displayedAnswers = ref<Record<number, string>>({})
let timer: number | undefined
let reconnectTimer: number | undefined
let streamAbort: AbortController | undefined
let typingFrame: number | undefined
let streamHealthy = false
const terminalStatuses = ['COMPLETED', 'PARTIAL', 'FAILED', 'CANCELLED', 'NEEDS_CLARIFICATION']
const typingTargets = new Map<number, string>()
const animatedMessageIds = new Set<number>()

function stopTypingAnimation() {
  if (typingFrame) window.cancelAnimationFrame(typingFrame)
  typingFrame = undefined
  typingTargets.clear()
  animatedMessageIds.clear()
}
function commonPrefixLength(left: string, right: string) {
  const limit = Math.min(left.length, right.length)
  let index = 0
  while (index < limit && left[index] === right[index]) index += 1
  return index
}
function typingStep() {
  typingFrame = undefined
  let hasPendingText = false
  let typingStateChanged = false
  for (const [id, target] of typingTargets.entries()) {
    const current = displayedAnswers.value[id] || ''
    if (current === target) {
      typingTargets.delete(id)
      const message = messages.value.find(item => item.id === id)
      if (message && terminalStatuses.includes(message.status)) {
        animatedMessageIds.delete(id)
        typingStateChanged = true
      }
      continue
    }
    const backlog = target.length - current.length
    const characterCount = backlog > 700 ? 8 : backlog > 350 ? 5 : backlog > 160 ? 3 : backlog > 80 ? 2 : 1
    displayedAnswers.value[id] = current + target.slice(current.length, current.length + characterCount)
    hasPendingText = true
  }
  if (typingStateChanged) displayedAnswers.value = { ...displayedAnswers.value }
  if (hasPendingText || typingTargets.size) typingFrame = window.requestAnimationFrame(typingStep)
}
function scheduleTyping() {
  if (!typingFrame && typingTargets.size) typingFrame = window.requestAnimationFrame(typingStep)
}
function applyMessages(data: any[], animate: boolean) {
  if (!animate) {
    stopTypingAnimation()
    displayedAnswers.value = Object.fromEntries(data.map(message => [message.id, message.answer || '']))
    messages.value = data
    return
  }
  const incomingIds = new Set(data.map(message => message.id))
  for (const key of Object.keys(displayedAnswers.value)) {
    if (!incomingIds.has(Number(key))) delete displayedAnswers.value[Number(key)]
  }
  messages.value = data
  for (const message of data) {
    const target = message.answer || ''
    const shouldStart = message.status === 'ANSWERING' && Boolean(target)
    const isAnimated = animatedMessageIds.has(message.id)
    if (!shouldStart && !isAnimated) {
      displayedAnswers.value[message.id] = target
      typingTargets.delete(message.id)
      continue
    }
    if (shouldStart && !isAnimated) {
      animatedMessageIds.add(message.id)
      displayedAnswers.value[message.id] = ''
    }
    const current = displayedAnswers.value[message.id] || ''
    if (!target.startsWith(current)) {
      const prefixLength = commonPrefixLength(current, target)
      if (!current.length || prefixLength / current.length < 0.8) {
        displayedAnswers.value[message.id] = target
        typingTargets.delete(message.id)
        if (terminalStatuses.includes(message.status)) animatedMessageIds.delete(message.id)
        continue
      }
      displayedAnswers.value[message.id] = target.slice(0, prefixLength)
    }
    typingTargets.set(message.id, target)
  }
  scheduleTyping()
}
function displayedAnswer(message: any) { return displayedAnswers.value[message.id] ?? message.answer ?? '' }
function isTyping(message: any) {
  return animatedMessageIds.has(message.id) && (message.status === 'ANSWERING' || typingTargets.has(message.id))
}

async function loadSessions() {
  sessions.value = (await api.get('/chat/sessions')).data
  if (!activeSession.value && sessions.value.length) await openSession(sessions.value[0].id)
}
async function newSession() {
  const { data } = await api.post('/chat/sessions', { title: '新对话' })
  sessions.value.unshift(data); activeSession.value = data.id; applyMessages([], false)
  startMessageStream(data.id)
}
async function openSession(id: number) {
  activeSession.value = id
  applyMessages((await api.get(`/chat/sessions/${id}/messages`)).data, false)
  startMessageStream(id)
}
async function refreshMessages() {
  const sessionId = activeSession.value
  if (!sessionId) return
  const data = (await api.get(`/chat/sessions/${sessionId}/messages`)).data
  if (activeSession.value === sessionId) applyMessages(data, true)
}
async function startMessageStream(id: number) {
  if (reconnectTimer) { window.clearTimeout(reconnectTimer); reconnectTimer = undefined }
  streamAbort?.abort()
  const controller = new AbortController()
  streamAbort = controller
  const baseURL = String(api.defaults.baseURL || '').replace(/\/$/, '')
  try {
    const response = await fetch(`${baseURL}/chat/sessions/${id}/events`, {
      headers: { Authorization: `Bearer ${localStorage.getItem('access_token') || ''}` },
      signal: controller.signal,
    })
    if (!response.ok || !response.body) throw new Error('SSE unavailable')
    streamHealthy = true
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (!controller.signal.aborted) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const blocks = buffer.split('\n\n')
      buffer = blocks.pop() || ''
      for (const block of blocks) {
        const dataLine = block.split('\n').find(line => line.startsWith('data: '))
        if (dataLine && activeSession.value === id) {
          try { applyMessages(JSON.parse(dataLine.slice(6)), true) } catch { /* wait for the next complete SSE event */ }
        }
      }
    }
  } catch (error: any) {
    if (error?.name !== 'AbortError') await refreshMessages()
  } finally {
    if (streamAbort === controller) {
      streamHealthy = false
      if (!controller.signal.aborted) {
        reconnectTimer = window.setTimeout(() => { if (activeSession.value === id) void startMessageStream(id) }, 3000)
      }
    }
  }
}
async function send() {
  if (sending.value || !question.value.trim()) return
  if (!activeSession.value) await newSession()
  sending.value = true
  try {
    const requestId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`
    const { data } = await api.post(`/chat/sessions/${activeSession.value}/messages`, { question: question.value }, { headers: { 'Idempotency-Key': requestId } })
    applyMessages([...messages.value, data], true); question.value = ''; await nextTick()
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '问题提交失败，请稍后重试')
  } finally { sending.value = false }
}
function replaceMessage(data: any) {
  const index = messages.value.findIndex(item => item.id === data.id)
  if (index >= 0) {
    const nextMessages = [...messages.value]
    nextMessages[index] = data
    applyMessages(nextMessages, true)
  }
}
function waitingSeconds(message: any) {
  return Math.max(0, Math.floor((Date.now() - new Date(message.created_at).getTime()) / 1000))
}
async function cancelMessage(message: any) {
  controlling.value = message.id
  try {
    const { data } = await api.post(`/chat/messages/${message.id}/cancel`)
    replaceMessage(data)
    ElMessage.success('任务已停止')
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '停止失败，正在刷新状态')
    await refreshMessages()
  } finally { controlling.value = null }
}
async function retryMessage(message: any) {
  controlling.value = message.id
  try {
    const { data } = await api.post(`/chat/messages/${message.id}/retry`)
    replaceMessage(data)
    ElMessage.success('已重新提交到问答队列')
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '重新执行失败')
  } finally { controlling.value = null }
}
async function resolveCompany(message: any, stockCode: string) {
  controlling.value = message.id
  try {
    const { data } = await api.post(`/chat/messages/${message.id}/resolve-company`, { stock_code: stockCode })
    replaceMessage(data)
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '公司确认失败，请刷新后重试')
    await refreshMessages()
  } finally { controlling.value = null }
}
function externalSource(url?: string) { return Boolean(url && /^https?:\/\//i.test(url)) }
function formatDate(value?: string) { return value ? value.slice(0, 16).replace('T', ' ') : '暂无可靠时间' }
function refreshTone(status?: string) {
  if (status === 'FAILED') return 'refresh-failed'
  if (status === 'PARTIAL') return 'refresh-partial'
  if (status === 'COMPLETED') return 'refresh-completed'
  return 'refresh-active'
}
function statusLabel(status?: string) {
  return ({ QUEUED: '排队中', RESOLVING_ENTITY: '识别标的', COLLECTING: '首次采集', PROCESSING: '检索分析', ANSWERING: '生成回答', NEEDS_CLARIFICATION: '需要确认', COMPLETED: '已完成', PARTIAL: '部分完成', FAILED: '失败', CANCELLED: '已停止' } as Record<string, string>)[status || ''] || status || '-'
}
function sourceLabel(source?: string) {
  return ({ RESEARCH_REPORT: '券商研报', NEWS: '财经新闻', ANNOUNCEMENT: '公司公告', SOCIAL: '公开舆情', MARKET_DATA: '行情数据' } as Record<string, string>)[source || ''] || source || '资料'
}
function resolvedSubject(message: any) { return message.analysis_metadata?.resolved_subject }
function contextResolution(message: any) {
  const value = message.analysis_metadata?.context_resolution
  if (!value?.standalone_question || value.standalone_question === value.original_question) return null
  return value
}
function answerModeLabel(message: any) {
  const mode = message.analysis_metadata?.answer_mode
  return ({
    SYSTEM_HELP: '系统帮助', CONVERSATION: '自然对话', GENERAL_KNOWLEDGE: '通用金融知识',
    CONTENT_GUIDANCE: '内容生成引导', OPEN_FALLBACK: '开放式回答', SCOPE_NOTICE: '能力范围说明',
    CLARIFICATION: '需要补充信息', LLM: '知识库证据回答', FILTERED_LLM: '知识库证据回答',
    LLM_WITH_WARNINGS: '证据辅助分析', EVIDENCE_FALLBACK: '证据降级回答', STRUCTURED_COMPARISON: '结构化机构对比',
  } as Record<string, string>)[mode || ''] || message.analysis_metadata?.research_intent_label || '综合研究'
}
function usesKnowledgeBase(message: any) {
  return Boolean(message.citations?.length || ['LLM', 'FILTERED_LLM', 'EVIDENCE_FALLBACK', 'STRUCTURED_COMPARISON'].includes(message.analysis_metadata?.answer_mode))
}
function llmDegraded(message: any) { return Boolean(message.analysis_metadata?.llm_call?.degraded) }
function renderMarkdown(content?: string) {
  if (!content) return ''
  return DOMPurify.sanitize(marked.parse(content, { async: false }) as string, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ['img', 'iframe', 'object', 'embed', 'script', 'style'],
  })
}
function progressSteps(message: any) {
  const steps = [
    { label: '理解上下文与问题', threshold: 7 },
    { label: '确认对象与知识库', threshold: 20 },
    { label: '按计划检索与调用工具', threshold: 55 },
    { label: '流式生成与证据校验', threshold: 80 },
  ]
  return steps.map((step, index) => ({ ...step, state: message.progress > step.threshold || terminalStatuses.includes(message.status) ? 'done' : message.progress >= step.threshold || (index === 0 && message.progress < 10) ? 'active' : 'pending' }))
}
function validationLabel(message: any) {
  const validation = message.analysis_metadata?.validation
  if (!validation) return '未记录校验'
  if (validation.mode === 'STRUCTURED_COMPARISON') return '结构化机构对比'
  if (validation.mode === 'EVIDENCE_FALLBACK') return '证据原文模式'
  if (validation.mode === 'FILTERED_LLM') return '已过滤无证据段落'
  if (validation.mode === 'LLM_WITH_WARNINGS') return '引用软校验'
  if (validation.mode === 'LLM_REPAIRED') return '自动修订后通过校验'
  return validation.valid ? '引用与数字校验通过' : '已降级为安全回答'
}
onMounted(async () => { await loadSessions(); if (!activeSession.value) await newSession(); timer = window.setInterval(() => { if (!streamHealthy) void refreshMessages() }, 10000) })
onUnmounted(() => { if (timer) window.clearInterval(timer); if (reconnectTimer) window.clearTimeout(reconnectTimer); streamAbort?.abort(); stopTypingAnimation() })
</script>

<template>
  <div class="chat-layout">
    <aside class="chat-sessions">
      <button class="primary-button" @click="newSession">＋ 新建研究对话</button>
      <div class="chat-session-list">
        <button v-for="item in sessions" :key="item.id" :class="{ active: activeSession === item.id }" @click="openSession(item.id)"><strong>{{ item.title }}</strong><small>{{ item.updated_at?.slice(0, 16).replace('T', ' ') }}</small></button>
      </div>
    </aside>
    <section class="chat-main">
      <div class="chat-header"><div><span class="eyebrow">LANGGRAPH AGENT</span><h1>智能研究问答</h1></div><span class="pill">自动检索 · 自动续答</span></div>
      <div class="message-list">
        <div v-if="!messages.length" class="chat-welcome"><span class="brand-mark large">F</span><h2>从公司或行业问题开始</h2><p>系统会自动判断问题类型；公司资料不足时自动采集，行业问题会跨公司检索现有知识库。</p><div class="suggestions"><button @click="question='宁德时代最近的机构观点有什么分歧？'">宁德时代的机构分歧</button><button @click="question='新能源汽车行业最近有哪些重要事件？'">新能源汽车行业</button><button @click="question='这个系统能做什么？'">查看系统能力</button></div></div>
        <article v-for="message in messages" :key="message.id" class="message-block">
          <div class="user-question"><span>你</span><p>{{ message.question }}</p></div>
          <div class="agent-answer">
            <span class="brand-mark small">F</span>
            <div v-if="!terminalStatuses.includes(message.status)" class="progress-answer">
              <div v-if="resolvedSubject(message)" class="entity-confirmation">
                <strong>已确认：{{ resolvedSubject(message).name }}</strong>
                <span v-if="resolvedSubject(message).stock_code">{{ resolvedSubject(message).stock_code }} · {{ resolvedSubject(message).exchange_label }} · {{ resolvedSubject(message).listing_status }}</span>
                <span v-else>{{ resolvedSubject(message).name }}</span>
              </div>
              <div v-if="contextResolution(message)" class="context-resolution"><strong>结合上文理解为：</strong>{{ contextResolution(message).standalone_question }}</div>
              <strong>{{ message.status_text }}</strong>
              <div class="process-steps"><span v-for="step in progressSteps(message)" :key="step.label" :class="`step-${step.state}`"><i></i>{{ step.label }}</span></div>
              <el-progress :percentage="message.progress" :stroke-width="8" />
              <div v-if="message.status === 'ANSWERING' && displayedAnswer(message)" class="streaming-draft" aria-live="polite">
                <div class="streaming-label"><span></span>AI 正在生成草稿 · 完成后会校验证据并可能调整内容</div>
                <div class="markdown-text" :class="{ typing: isTyping(message) }" v-html="renderMarkdown(displayedAnswer(message))"></div>
              </div>
              <div class="task-controls"><small>{{ statusLabel(message.status) }} · 已等待 {{ waitingSeconds(message) }} 秒 · 状态会自动更新</small><button class="stop-button" :disabled="controlling === message.id" @click="cancelMessage(message)">停止任务</button></div>
              <p v-if="waitingSeconds(message) >= 30" class="queue-warning">等待时间较长，可停止后重新执行；问答队列不会再被采集任务阻塞。</p>
            </div>
            <div v-else-if="message.status === 'FAILED'" class="error-box"><p>{{ message.error || message.status_text }}</p><button class="retry-button" :disabled="controlling === message.id" @click="retryMessage(message)">重新执行</button></div>
            <div v-else-if="message.status === 'CANCELLED'" class="cancelled-box"><p>{{ message.status_text }}</p><button class="retry-button" :disabled="controlling === message.id" @click="retryMessage(message)">重新执行</button></div>
            <div v-else-if="message.status === 'NEEDS_CLARIFICATION'" class="clarification-box"><strong>{{ message.status_text }}</strong><p>{{ message.answer || '请补充准确标的，系统不会在公司不明确时猜测。' }}</p><div v-if="message.clarification_candidates?.length" class="candidate-buttons"><button v-for="candidate in message.clarification_candidates || []" :key="candidate.stock_code" :disabled="controlling === message.id" @click="resolveCompany(message, candidate.stock_code)">{{ candidate.name }}（{{ candidate.stock_code }} · {{ candidate.exchange_label || candidate.exchange }}）</button></div><small v-else>可以直接在下方输入公司名称、6位股票代码或更完整的问题。</small></div>
            <div v-else class="answer-content">
              <div v-if="resolvedSubject(message)" class="entity-confirmation compact"><strong>{{ resolvedSubject(message).name }}<template v-if="resolvedSubject(message).stock_code">（{{ resolvedSubject(message).stock_code }}）</template></strong><span>{{ resolvedSubject(message).exchange_label }} · {{ resolvedSubject(message).listing_status }}</span></div>
              <div class="answer-meta"><span :class="`status-${message.status.toLowerCase()}`">{{ statusLabel(message.status) }}</span><span>{{ answerModeLabel(message) }}</span><span>可信度 {{ message.confidence || '-' }}</span><span v-if="usesKnowledgeBase(message)">资料截至 {{ formatDate(message.data_as_of) }}</span><span v-if="usesKnowledgeBase(message)">{{ validationLabel(message) }}</span></div>
              <div v-if="contextResolution(message)" class="context-resolution compact"><strong>本轮理解：</strong>{{ contextResolution(message).standalone_question }}</div>
              <div v-if="message.refresh_status" class="refresh-notice" :class="refreshTone(message.refresh_status)"><strong>后台资料更新 · {{ statusLabel(message.refresh_status) }}</strong><span>{{ message.refresh_status_text }}</span><small v-if="['QUEUED', 'RUNNING'].includes(message.refresh_status)">当前回答已经完成，无需等待；更新完成后会自动进入共享知识库，供后续问题使用。</small></div>
              <div v-if="llmDegraded(message)" class="partial-warning">远程对话模型本次未返回可用结果，系统已切换为安全兜底。<button class="retry-button" :disabled="controlling === message.id" @click="retryMessage(message)">重新执行</button></div>
              <div v-if="message.analysis_metadata?.validation?.mode === 'LLM_WITH_WARNINGS'" class="analysis-note">回答包含模型基于证据作出的归纳、计算或推理，不代表来源原文逐字表述；引用编号与交易安全已校验，具体事实和数字请结合展开的来源核对。</div>
              <div v-if="message.analysis_metadata?.retrieval_scope?.recency_fallback_used" class="partial-warning">近180天内相关证据不足，本次使用了更早的历史资料，请注意资料日期。</div>
              <div v-if="message.analysis_metadata?.retrieval_scope?.intent_fallback_used" class="partial-warning">首选资料类型不足，本次已回退检索该公司的其他资料，结论可能不够完整。</div>
              <div v-if="message.analysis_metadata?.failed_tools?.length" class="partial-warning">部分分析工具暂时不可用，回答已使用其余有效证据降级完成。</div>
              <div v-if="message.missing_sources?.length" class="partial-warning">本次回答缺少部分来源：{{ message.missing_sources.map(sourceLabel).join('、') }}</div>
              <div class="markdown-text" :class="{ typing: isTyping(message) }" v-html="renderMarkdown(displayedAnswer(message))"></div>
              <details v-if="message.citations?.length"><summary>查看 {{ message.citations.length }} 条证据来源</summary><component :is="externalSource(citation.source_url) ? 'a' : 'div'" v-for="(citation, index) in message.citations" :key="index" :href="externalSource(citation.source_url) ? citation.source_url : undefined" target="_blank" rel="noopener" class="citation"><strong>[{{ index + 1 }}] {{ citation.title }}</strong><p>{{ citation.quote }}</p><small>{{ sourceLabel(citation.source_type) }} · {{ citation.page ? `第${citation.page}页` : citation.source_type === 'MARKET_DATA' ? '结构化行情' : '网页正文' }} · {{ externalSource(citation.source_url) ? '打开原文 ↗' : citation.source_name || '本地演示快照' }}</small></component></details>
            </div>
          </div>
        </article>
      </div>
      <form class="chat-composer" @submit.prevent="send"><textarea v-model="question" placeholder="输入公司、行业、研报、财务或风险问题…" @keydown.ctrl.enter="send"></textarea><button class="primary-button inline" :disabled="sending">发送</button><small>Ctrl + Enter 发送 · 重要结论将附原文来源</small></form>
    </section>
  </div>
</template>
