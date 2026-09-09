<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { use } from 'echarts/core'
import { BarChart, CandlestickChart, LineChart, PieChart } from 'echarts/charts'
import { DataZoomComponent, GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { ECharts } from 'echarts/core'
import { init } from 'echarts/core'
import { api } from '../api'

const route = useRoute()
const overview = ref<any>(null)
const timeline = ref<any[]>([])
const sentiment = ref<any>({ distribution: {} })
const knowledge = ref<any>(null)
const researchMetrics = ref<any>({ forecasts: [], ratings: [], sentiments: [] })
const events = ref<any[]>([])
const market = ref<any>({ status: 'EMPTY', items: [] })
const loading = ref(true)
const documentChart = ref<HTMLElement | null>(null)
const sentimentChart = ref<HTMLElement | null>(null)
const forecastChart = ref<HTMLElement | null>(null)
const ratingChart = ref<HTMLElement | null>(null)
const priceChart = ref<HTMLElement | null>(null)
const sentimentLabels: Record<string, string> = {
  POSITIVE: '积极',
  SLIGHTLY_POSITIVE: '偏积极',
  NEUTRAL: '中性',
  SLIGHTLY_NEGATIVE: '偏消极',
  NEGATIVE: '消极',
}

function sentimentLabel(value: string) {
  return sentimentLabels[String(value).toUpperCase()] || '未分类'
}
const documentVisible = ref(false)
const documentLoading = ref(false)
const selectedDocument = ref<any>(null)
use([BarChart, CandlestickChart, LineChart, PieChart, DataZoomComponent, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])
let charts: ECharts[] = []
const typeName: Record<string, string> = { RESEARCH_REPORT: '研报', NEWS: '新闻', ANNOUNCEMENT: '公告', SOCIAL: '舆情' }

function isExternalSource(url?: string) {
  return Boolean(url && /^https?:\/\//i.test(url))
}

function formattedDocumentBody(text?: string) {
  if (!text) return '<p>当前资料没有可展示的解析正文。</p>'
  return DOMPurify.sanitize(marked.parse(text, { breaks: true, gfm: true }) as string)
}

async function openDocument(document: any) {
  documentVisible.value = true
  documentLoading.value = true
  selectedDocument.value = null
  try {
    selectedDocument.value = (await api.get(`/documents/${document.id}`)).data
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '资料详情加载失败')
    documentVisible.value = false
  } finally {
    documentLoading.value = false
  }
}

async function load() {
  loading.value = true
  const code = route.params.code
  try {
    const [a, b, c, d, e, f, g] = await Promise.all([api.get(`/companies/${code}/overview`), api.get(`/companies/${code}/timeline`), api.get(`/analysis/${code}/sentiment`), api.get(`/companies/${code}/knowledge-status`), api.get(`/companies/${code}/research-metrics`), api.get(`/companies/${code}/events`), api.get(`/companies/${code}/market-history`)])
    overview.value = a.data
    timeline.value = b.data
    sentiment.value = c.data
    knowledge.value = d.data
    researchMetrics.value = e.data
    events.value = f.data
    market.value = g.data
    await nextTick()
    renderCharts()
  } finally {
    loading.value = false
  }
}

function renderCharts() {
  charts.forEach(chart => chart.dispose())
  charts = []
  if (documentChart.value) {
    const chart = init(documentChart.value)
    charts.push(chart)
    const entries = Object.entries(overview.value?.document_counts || {})
    chart.setOption({ tooltip: {}, grid: { left: 44, right: 15, top: 25, bottom: 35 }, xAxis: { type: 'category', data: entries.map(([key]) => typeName[key] || key) }, yAxis: { type: 'value', minInterval: 1 }, series: [{ type: 'bar', data: entries.map(([, value]) => value), itemStyle: { color: '#6f9537', borderRadius: [6, 6, 0, 0] } }] })
  }
  if (sentimentChart.value) {
    const chart = init(sentimentChart.value)
    charts.push(chart)
    const entries = Object.entries(sentiment.value?.distribution || {})
    chart.setOption({ tooltip: { trigger: 'item' }, legend: { bottom: 0 }, series: [{ type: 'pie', radius: ['42%', '68%'], data: entries.map(([name, value]) => ({ name: sentimentLabel(name), value })), itemStyle: { borderColor: '#fff', borderWidth: 2 } }] })
  }
  if (forecastChart.value) {
    const chart = init(forecastChart.value)
    charts.push(chart)
    const grouped: Record<string, number[]> = {}
    for (const item of researchMetrics.value.forecasts || []) {
      if (item.net_profit != null) (grouped[String(item.year)] ||= []).push(Number(item.net_profit))
    }
    const years = Object.keys(grouped).sort()
    chart.setOption({ tooltip: { trigger: 'axis' }, grid: { left: 50, right: 18, top: 28, bottom: 35 }, xAxis: { type: 'category', data: years }, yAxis: { type: 'value', name: '机构均值' }, series: [{ type: 'line', smooth: true, data: years.map(year => grouped[year].reduce((a,b) => a+b, 0) / grouped[year].length), symbolSize: 8, lineStyle: { color: '#779d36' }, itemStyle: { color: '#779d36' }, areaStyle: { color: '#dce9bd66' } }] })
  }
  if (ratingChart.value) {
    const chart = init(ratingChart.value)
    charts.push(chart)
    const counts: Record<string, number> = {}
    for (const item of researchMetrics.value.ratings || []) counts[item.rating || '未提取'] = (counts[item.rating || '未提取'] || 0) + 1
    chart.setOption({ tooltip: { trigger: 'item' }, legend: { bottom: 0 }, series: [{ type: 'pie', radius: ['38%', '68%'], data: Object.entries(counts).map(([name, value]) => ({ name, value })) }] })
  }
  if (priceChart.value && market.value.items?.length) {
    const chart = init(priceChart.value)
    charts.push(chart)
    const rows = market.value.items
    chart.setOption({ tooltip: { trigger: 'axis' }, grid: { left: 55, right: 20, top: 24, bottom: 55 }, xAxis: { type: 'category', data: rows.map((item: any) => item.date?.slice(0, 10)), boundaryGap: true }, yAxis: { scale: true }, dataZoom: [{ type: 'inside', start: 35, end: 100 }, { type: 'slider', start: 35, end: 100, height: 18, bottom: 8 }], series: [{ name: '日K', type: 'candlestick', data: rows.map((item: any) => [item.open, item.close, item.low, item.high]), itemStyle: { color: '#d85f50', color0: '#4d9b71', borderColor: '#d85f50', borderColor0: '#4d9b71' } }] })
  }
}

onMounted(load)
onUnmounted(() => charts.forEach(chart => chart.dispose()))
watch(() => route.params.code, load)
</script>

<template>
  <div class="page-wrap" v-loading="loading">
    <template v-if="overview">
      <div class="page-heading company-heading"><div><span class="eyebrow">{{ overview.company.stock_code }} · {{ overview.company.exchange }}</span><h1>{{ overview.company.name }}</h1><p>{{ overview.company.full_name }} · {{ overview.company.industry }}</p></div><span class="track-tag" :class="overview.company.tracking_mode.toLowerCase()">{{ overview.company.tracking_mode }}</span></div>
      <div class="metric-row">
        <article v-for="(count, type) in overview.document_counts" :key="type"><span>{{ typeName[String(type)] || type }}</span><strong>{{ count }}</strong><small>篇已入库资料</small></article>
      </div>
      <section class="knowledge-strip"><span :class="knowledge?.is_fresh ? 'health-ok' : 'health-warning'">{{ knowledge?.is_fresh ? '资料较新' : '资料可能需要更新' }}</span><span>共 {{ knowledge?.total_documents || 0 }} 篇资料</span><span>{{ knowledge?.indexed_chunks || 0 }} 个检索片段</span><span>在线 {{ overview.acquisition_counts?.LIVE || 0 }} · 快照 {{ overview.acquisition_counts?.SNAPSHOT || 0 }}</span><span v-if="knowledge?.is_collecting">正在采集</span></section>
      <section class="section-card"><div class="section-title"><div><h2>日线行情背景</h2><p>前复权日K · {{ market.source_name || '公开行情源' }}；仅用于辅助理解研报所处市场背景，不用于预测或交易</p></div><span v-if="market.data_as_of" class="pill">截至 {{ market.data_as_of.slice(0,10) }}</span></div><div v-if="market.items?.length" ref="priceChart" class="price-chart"></div><div v-else class="empty-state compact">尚未同步行情；管理员可在知识库后台触发同步。</div></section>
      <div class="chart-grid"><section class="section-card"><div class="section-title"><div><h2>资料构成</h2><p>四类知识来源数量</p></div></div><div ref="documentChart" class="chart"></div></section><section class="section-card"><div class="section-title"><div><h2>情感分布</h2><p>研报、新闻与舆情倾向</p></div></div><div ref="sentimentChart" class="chart"></div></section></div>
      <div class="chart-grid"><section class="section-card"><div class="section-title"><div><h2>净利润预测趋势</h2><p>按年份汇总机构预测均值，仅用于研究比较</p></div></div><div v-if="researchMetrics.forecasts?.length" ref="forecastChart" class="chart"></div><div v-else class="empty-state compact">暂无结构化盈利预测</div></section><section class="section-card"><div class="section-title"><div><h2>评级分布</h2><p>当前知识库已提取的券商评级</p></div></div><div v-if="researchMetrics.ratings?.length" ref="ratingChart" class="chart"></div><div v-else class="empty-state compact">暂无结构化评级</div></section></div>
      <div class="two-column">
        <section class="section-card">
          <div class="section-title"><div><h2>最新资料</h2><p>点击可在系统内查看解析正文和原始来源。</p></div></div>
          <div v-if="!overview.latest_documents.length" class="empty-state">暂无资料，可在问答或管理后台触发自动采集。</div>
          <button v-for="doc in overview.latest_documents" :key="doc.id" type="button" class="document-row document-button" @click="openDocument(doc)">
            <span class="doc-type">{{ typeName[doc.document_type] || doc.document_type }}</span><div><strong>{{ doc.title }}</strong><small>{{ doc.source_name }} · {{ doc.published_at?.slice(0, 10) || '时间未知' }} · {{ doc.acquisition_mode === 'SNAPSHOT' ? '演示快照' : '在线采集' }}</small></div><span>查看</span>
          </button>
        </section>
        <section class="section-card">
          <div class="section-title"><div><h2>事件聚类</h2><p>相似报道合并展示，避免重复新闻放大影响</p></div></div>
          <div class="timeline">
            <div v-for="item in events" :key="item.event_id" class="timeline-item"><i></i><div><small>{{ item.date?.slice(0, 10) || '时间未知' }} · {{ item.document_count }}篇资料 · {{ item.source_count }}个来源</small><strong>{{ item.title }}</strong><p>{{ item.summary?.slice(0, 120) }}</p></div></div>
          </div>
        </section>
      </div>
    </template>

    <el-dialog v-model="documentVisible" width="820px" top="5vh" destroy-on-close>
      <template #header><strong>{{ selectedDocument?.title || '资料详情' }}</strong></template>
      <div v-loading="documentLoading" class="document-detail">
        <template v-if="selectedDocument">
          <div class="document-meta"><span class="doc-type">{{ typeName[selectedDocument.document_type] || selectedDocument.document_type }}</span><span>{{ selectedDocument.source_name }}</span><span>{{ selectedDocument.published_at?.slice(0, 10) || '时间未知' }}</span><span>{{ selectedDocument.status }}</span><span :class="['source-mode', selectedDocument.acquisition_mode?.toLowerCase()]">{{ selectedDocument.acquisition_mode === 'SNAPSHOT' ? '演示快照' : '在线采集' }}</span><span v-if="selectedDocument.parse_quality != null" :class="selectedDocument.parse_quality >= 0.7 ? 'health-ok' : 'health-warning'">解析质量 {{ Math.round(selectedDocument.parse_quality * 100) }}%</span></div>
          <div v-if="selectedDocument.parse_warnings?.length" class="parse-warning"><strong>解析提示</strong><span>{{ selectedDocument.parse_warnings.join('；') }}</span></div>
          <section v-if="selectedDocument.summary"><h3>{{ selectedDocument.summary_method === 'LLM' ? '智能摘要' : '正文预览' }}</h3><small v-if="selectedDocument.summary_method !== 'LLM'" class="summary-note">本条资料未获得有效的大模型摘要，以下为清洗后的正文预览，不代表模型结论。</small><p>{{ selectedDocument.summary }}</p></section>
          <section v-if="selectedDocument.structured?.rating"><h3>投资评级</h3><div class="structured-card"><strong>{{ selectedDocument.structured.rating.institution || selectedDocument.source_name }}</strong><span>{{ selectedDocument.structured.rating.rating || '未提取' }}</span><span>目标价 {{ selectedDocument.structured.rating.target_price ?? '未披露' }}</span></div></section>
          <section v-if="selectedDocument.structured?.forecasts?.length"><h3>盈利预测</h3><el-table :data="selectedDocument.structured.forecasts" size="small"><el-table-column prop="year" label="年度" /><el-table-column prop="revenue" label="收入" /><el-table-column prop="net_profit" label="净利润" /><el-table-column prop="eps" label="EPS" /><el-table-column prop="unit" label="单位" /></el-table></section>
          <section v-if="selectedDocument.structured?.opinions?.length"><h3>核心观点</h3><ul><li v-for="(item,index) in selectedDocument.structured.opinions" :key="index">{{ item.content }}</li></ul></section>
          <section v-if="selectedDocument.structured?.risks?.length"><h3>风险提示</h3><ul><li v-for="(item,index) in selectedDocument.structured.risks" :key="index"><strong>{{ item.category }}</strong>：{{ item.content }}</li></ul></section>
          <details v-if="selectedDocument.evidences?.length" class="evidence-panel"><summary>查看 {{ selectedDocument.evidences.length }} 条结构化字段原文证据</summary><blockquote v-for="(item,index) in selectedDocument.evidences" :key="index"><small>{{ item.entity_type }}{{ item.page ? ` · 第${item.page}页` : '' }}</small><p>{{ item.quote }}</p></blockquote></details>
          <section><h3>格式化正文</h3><div class="formatted-document" v-html="formattedDocumentBody(selectedDocument.parsed_text)"></div></section>
          <div class="document-actions">
            <a v-if="isExternalSource(selectedDocument.source_url)" class="primary-button inline" :href="selectedDocument.source_url" target="_blank" rel="noopener">打开原始来源 ↗</a>
            <span v-else class="snapshot-note">这是本地演示快照，暂无外部原文地址。</span>
          </div>
        </template>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.document-button { width: 100%; border-right: 0; border-bottom: 0; border-left: 0; background: transparent; text-align: left; }
.document-button:hover { background: #f7f9f5; }
.document-detail { min-height: 180px; }
.document-meta { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; color: #728078; font-size: 12px; }
.document-detail section { margin-top: 20px; }
.document-detail h3 { margin: 0 0 10px; color: #18382e; }
.document-detail p { line-height: 1.75; }
.formatted-document { max-height: 52vh; overflow: auto; padding: 18px; border-radius: 10px; background: #f6f8f4; word-break: break-word; font: 13px/1.75 Inter, "Microsoft YaHei", sans-serif; }
.formatted-document :deep(p) { margin: 0 0 12px; }
.formatted-document :deep(table) { width: 100%; margin: 12px 0; border-collapse: collapse; background: #fff; }
.formatted-document :deep(th), .formatted-document :deep(td) { padding: 8px 10px; border: 1px solid #dfe5dc; text-align: left; vertical-align: top; }
.formatted-document :deep(th) { position: sticky; top: 0; background: #edf2e8; color: #18382e; }
.parse-warning { display: flex; gap: 10px; margin-top: 14px; padding: 10px 12px; border: 1px solid #e7c877; border-radius: 8px; background: #fff9e9; color: #775b17; font-size: 13px; }
.summary-note { display: block; margin: -4px 0 8px; color: #8a7650; }
.document-actions { display: flex; align-items: center; justify-content: flex-end; margin-top: 18px; }
.snapshot-note { color: #7b877f; font-size: 13px; }
.empty-state.compact { padding: 80px 20px; }
.structured-card { display:flex;gap:16px;align-items:center;padding:13px;background:#f6f8f4;border-radius:9px; }
.evidence-panel { margin-top:20px;padding:14px;background:#f8faf6;border-radius:9px; }
.evidence-panel blockquote { margin:12px 0 0;padding:10px 14px;border-left:3px solid #a9c93c;background:white; }
.evidence-panel blockquote p { margin:5px 0 0; }
</style>
