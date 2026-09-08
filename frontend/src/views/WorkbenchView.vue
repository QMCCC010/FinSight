<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '../api'
import type { Company, ReportItem, ReportMaterial, ReportVersion } from '../types'

const SOURCE_LABELS: Record<string, string> = {
  RESEARCH_REPORT: '券商研报',
  NEWS: '财经新闻',
  ANNOUNCEMENT: '公司公告',
  SOCIAL: '社交舆情',
}
const STATUS_LABELS: Record<string, string> = {
  QUEUED: '排队中', RUNNING: '生成中', COMPLETED: '已完成', PARTIAL: '部分完成', FAILED: '生成失败', CANCELLED: '已停止',
}
const companies = ref<Company[]>([])
const reports = ref<ReportItem[]>([])
const selected = ref<ReportItem | null>(null)
const materials = ref<ReportMaterial | null>(null)
const versions = ref<ReportVersion[]>([])
const selectedDocumentIds = ref<number[]>([])
const form = ref({
  stock_code: '002594',
  report_type: 'COMPANY_BRIEF',
  source_types: ['RESEARCH_REPORT', 'NEWS', 'ANNOUNCEMENT', 'SOCIAL'],
})
const dateRange = ref<[Date, Date] | null>(null)
const generating = ref(false)
const loadingMaterials = ref(false)
const refreshingMaterials = ref(false)
const materialSearch = ref('')
const editMode = ref(false)
const draftTitle = ref('')
const draftContent = ref('')
const rightTab = ref<'materials' | 'citations' | 'versions'>('materials')
let materialTimer: number | undefined
let pollTimer: number | undefined

const activeReport = computed(() => selected.value && ['QUEUED', 'RUNNING'].includes(selected.value.status))
const visibleMaterials = computed(() => {
  const keyword = materialSearch.value.trim().toLowerCase()
  return (materials.value?.documents || []).filter(item => !keyword || `${item.title}${item.source_name}`.toLowerCase().includes(keyword))
})
const canGenerate = computed(() => {
  if (generating.value || loadingMaterials.value) return false
  if (form.value.report_type === 'REPORT_COMPARISON') return selectedDocumentIds.value.length >= 2
  return Boolean(form.value.source_types.length && materials.value?.total)
})

function dateParams() {
  if (!dateRange.value) return {}
  const from = new Date(dateRange.value[0]); from.setHours(0, 0, 0, 0)
  const to = new Date(dateRange.value[1]); to.setHours(23, 59, 59, 999)
  return { date_from: from.toISOString(), date_to: to.toISOString() }
}

async function load() {
  try {
    const [companyResponse, reportResponse] = await Promise.all([api.get('/companies?tracked_only=true'), api.get('/reports')])
    companies.value = companyResponse.data
    reports.value = reportResponse.data
    if (!companies.value.some(item => item.stock_code === form.value.stock_code) && companies.value.length) form.value.stock_code = companies.value[0].stock_code
    if (reports.value.length) await chooseReport(reports.value[0])
    await loadMaterials(true)
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '内容工作台加载失败')
  }
}

async function loadMaterials(resetSelection = false) {
  if (!form.value.stock_code) return
  loadingMaterials.value = true
  try {
    const response = await api.get('/reports/materials', {
      params: {
        stock_code: form.value.stock_code,
        report_type: form.value.report_type,
        source_types: form.value.report_type === 'REPORT_COMPARISON' ? ['RESEARCH_REPORT'] : form.value.source_types,
        ...dateParams(),
      },
    })
    materials.value = response.data
    const validIds = new Set(response.data.documents.map((item: any) => item.id))
    selectedDocumentIds.value = selectedDocumentIds.value.filter(id => validIds.has(id))
    if (resetSelection || form.value.report_type === 'REPORT_COMPARISON' && selectedDocumentIds.value.length < 2) {
      selectedDocumentIds.value = response.data.documents.filter((item: any) => item.usable).slice(0, form.value.report_type === 'REPORT_COMPARISON' ? 5 : 30).map((item: any) => item.id)
    }
  } catch (error: any) {
    materials.value = null
    ElMessage.error(error.response?.data?.detail || '读取知识库资料失败')
  } finally {
    loadingMaterials.value = false
  }
}

function scheduleMaterials() {
  window.clearTimeout(materialTimer)
  materialTimer = window.setTimeout(() => loadMaterials(true), 250)
}

async function generate() {
  if (!canGenerate.value) return
  generating.value = true
  try {
    const payload = {
      ...form.value,
      source_types: form.value.report_type === 'REPORT_COMPARISON' ? ['RESEARCH_REPORT'] : form.value.source_types,
      document_ids: selectedDocumentIds.value.length ? selectedDocumentIds.value : undefined,
      ...dateParams(),
    }
    const report = (await api.post('/reports', payload)).data as ReportItem
    reports.value.unshift(report)
    await chooseReport(report)
    startPolling()
    ElMessage.success('报告已进入后台生成队列')
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '创建报告失败')
  } finally {
    generating.value = false
  }
}

async function refreshMaterials() {
  if (!form.value.stock_code) return
  refreshingMaterials.value = true
  try {
    const response = await api.post(`/reports/materials/${form.value.stock_code}/refresh`, {
      source_types: form.value.report_type === 'REPORT_COMPARISON' ? ['RESEARCH_REPORT'] : form.value.source_types,
    })
    ElMessage.success(response.data.reused ? '该公司已有后台更新任务，已复用' : '资料已进入后台更新队列，可继续使用当前知识库生成')
    await loadMaterials(false)
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '提交后台更新失败')
  } finally {
    refreshingMaterials.value = false
  }
}

async function chooseReport(report: ReportItem) {
  editMode.value = false
  selected.value = report
  draftTitle.value = report.title
  draftContent.value = report.content_markdown || ''
  versions.value = []
  if (['COMPLETED', 'PARTIAL'].includes(report.status)) {
    versions.value = (await api.get(`/reports/${report.id}/versions`)).data
  }
  if (['QUEUED', 'RUNNING'].includes(report.status)) startPolling()
}

function startPolling() {
  window.clearInterval(pollTimer)
  pollTimer = window.setInterval(refreshSelected, 1500)
}

async function refreshSelected() {
  if (!selected.value || !['QUEUED', 'RUNNING'].includes(selected.value.status)) {
    window.clearInterval(pollTimer)
    return
  }
  try {
    const fresh = (await api.get(`/reports/${selected.value.id}`)).data as ReportItem
    selected.value = fresh
    const index = reports.value.findIndex(item => item.id === fresh.id)
    if (index >= 0) reports.value[index] = fresh
    if (!['QUEUED', 'RUNNING'].includes(fresh.status)) {
      window.clearInterval(pollTimer)
      draftTitle.value = fresh.title
      draftContent.value = fresh.content_markdown || ''
      if (['COMPLETED', 'PARTIAL'].includes(fresh.status)) versions.value = (await api.get(`/reports/${fresh.id}/versions`)).data
    }
  } catch {
    window.clearInterval(pollTimer)
  }
}

async function cancelReport() {
  if (!selected.value) return
  const updated = (await api.post(`/reports/${selected.value.id}/cancel`)).data as ReportItem
  selected.value = updated
  await refreshHistoryItem(updated)
}

async function retryReport() {
  if (!selected.value) return
  const updated = (await api.post(`/reports/${selected.value.id}/retry`)).data as ReportItem
  selected.value = updated
  await refreshHistoryItem(updated)
  startPolling()
}

async function saveReport() {
  if (!selected.value) return
  try {
    const updated = (await api.patch(`/reports/${selected.value.id}`, { title: draftTitle.value, content_markdown: draftContent.value })).data
    selected.value = updated
    await refreshHistoryItem(updated)
    versions.value = (await api.get(`/reports/${updated.id}/versions`)).data
    editMode.value = false
    ElMessage.success('报告已保存为新版本')
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '保存失败')
  }
}

async function deleteReport() {
  if (!selected.value) return
  await ElMessageBox.confirm('删除后该报告将从“我的报告”中移除，确定继续吗？', '删除报告', { type: 'warning' })
  const id = selected.value.id
  await api.delete(`/reports/${id}`)
  reports.value = reports.value.filter(item => item.id !== id)
  selected.value = reports.value[0] || null
  if (selected.value) await chooseReport(selected.value)
  ElMessage.success('报告已删除')
}

async function refreshHistoryItem(report: ReportItem) {
  const index = reports.value.findIndex(item => item.id === report.id)
  if (index >= 0) reports.value[index] = report
}

async function copyReport() {
  if (!selected.value?.content_markdown) return
  await navigator.clipboard.writeText(selected.value.content_markdown)
  ElMessage.success('报告Markdown已复制')
}

async function download() {
  if (!selected.value) return
  const response = await api.get(`/reports/${selected.value.id}/download`, { responseType: 'blob' })
  const url = URL.createObjectURL(response.data)
  const link = document.createElement('a')
  link.href = url; link.download = `report-${selected.value.id}.md`; link.click()
  URL.revokeObjectURL(url)
}

async function exportReport(format: 'docx' | 'pdf') {
  if (!selected.value) return
  try {
    const response = await api.get(`/reports/${selected.value.id}/export`, { params: { format }, responseType: 'blob', timeout: 120000 })
    const url = URL.createObjectURL(response.data)
    const link = document.createElement('a')
    link.href = url; link.download = `report-${selected.value.id}.${format}`; link.click()
    URL.revokeObjectURL(url)
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || `导出${format.toUpperCase()}失败`)
  }
}

function selectVisible() {
  selectedDocumentIds.value = [...new Set([...selectedDocumentIds.value, ...visibleMaterials.value.filter(item => item.usable).map(item => item.id)])].slice(0, 50)
}

function restoreVersion(version: ReportVersion) {
  draftTitle.value = version.title
  draftContent.value = version.content_markdown
  editMode.value = true
  ElMessage.info(`已载入版本 V${version.version_number}，保存后会创建新版本`)
}

watch(() => [form.value.stock_code, form.value.report_type, [...form.value.source_types], dateRange.value], () => {
  if (form.value.report_type === 'REPORT_COMPARISON' && (form.value.source_types.length !== 1 || form.value.source_types[0] !== 'RESEARCH_REPORT')) form.value.source_types = ['RESEARCH_REPORT']
  scheduleMaterials()
}, { deep: true })

onMounted(load)
onBeforeUnmount(() => { window.clearTimeout(materialTimer); window.clearInterval(pollTimer) })
</script>

<template>
  <div class="page-wrap workbench-page">
    <div class="page-heading">
      <div><span class="eyebrow">RESEARCH CONTENT STUDIO</span><h1>内容生成工作台</h1><p>从资料选择、证据化生成到编辑和版本管理的一体化研究流程。</p></div>
    </div>

    <div class="studio-grid">
      <aside class="section-card studio-settings">
        <h2>报告设置</h2>
        <label>分析公司
          <el-select v-model="form.stock_code" filterable><el-option v-for="item in companies" :key="item.id" :label="`${item.name} · ${item.stock_code}`" :value="item.stock_code" /></el-select>
        </label>
        <label>报告模板
          <el-select v-model="form.report_type"><el-option label="公司研究简报" value="COMPANY_BRIEF" /><el-option label="多研报观点对比" value="REPORT_COMPARISON" /></el-select>
        </label>
        <p class="template-description" v-if="form.report_type === 'COMPANY_BRIEF'">综合财务实际值、机构预测、评级、事件、舆情和风险。</p>
        <p class="template-description" v-else>只比较所选券商研报，识别评级、预测、共识、分歧和风险差异。</p>
        <label>资料时间范围<el-date-picker v-model="dateRange" type="daterange" start-placeholder="开始日期" end-placeholder="结束日期" /></label>
        <label v-if="form.report_type === 'COMPANY_BRIEF'">信息来源
          <el-checkbox-group v-model="form.source_types"><el-checkbox value="RESEARCH_REPORT">研报</el-checkbox><el-checkbox value="NEWS">新闻</el-checkbox><el-checkbox value="ANNOUNCEMENT">公告</el-checkbox><el-checkbox value="SOCIAL">舆情</el-checkbox></el-checkbox-group>
        </label>
        <div v-if="materials" class="coverage-card">
          <div><strong>{{ materials.total }}</strong><small>匹配资料</small></div>
          <div><strong>{{ materials.usable }}</strong><small>完成索引</small></div>
          <p>最近采集：{{ materials.last_crawled_at?.slice(0, 16).replace('T', ' ') || '暂无记录' }}</p>
          <div class="freshness-list"><span v-for="item in materials.freshness" :key="item.label" :class="item.is_fresh ? 'fresh' : 'stale'">{{ item.label }} {{ item.count }}</span></div>
          <p v-for="warning in materials.warnings" :key="warning" class="coverage-warning">{{ warning }}</p>
          <button class="refresh-materials-button" :disabled="refreshingMaterials || materials.is_collecting" @click="refreshMaterials">{{ materials.is_collecting ? '资料正在后台更新' : refreshingMaterials ? '正在提交…' : '后台更新资料' }}</button>
        </div>
        <button class="primary-button" :disabled="!canGenerate" @click="generate">{{ generating ? '正在提交…' : '生成研究报告' }}</button>

        <div class="history">
          <h3>我的报告</h3>
          <div v-if="!reports.length" class="mini-empty">尚未生成报告</div>
          <button v-for="item in reports" :key="item.id" :class="{ active: selected?.id === item.id }" @click="chooseReport(item)">
            <strong>{{ item.title }}</strong><small>{{ item.created_at?.slice(0, 10) }} · {{ STATUS_LABELS[item.status] || item.status }}</small>
          </button>
        </div>
      </aside>

      <main class="section-card studio-editor">
        <div class="section-title">
          <div><h2>{{ selected?.title || '报告编辑与预览' }}</h2><p v-if="selected">{{ STATUS_LABELS[selected.status] || selected.status }} · {{ versions.length }} 个版本</p><p v-else>生成或选择一份报告后在这里查看。</p></div>
          <div v-if="selected && ['COMPLETED', 'PARTIAL'].includes(selected.status)" class="button-group">
            <button v-if="!editMode" class="secondary-button" @click="editMode=true">编辑</button>
            <button v-else class="secondary-button" @click="saveReport">保存新版本</button>
            <button class="secondary-button" @click="copyReport">复制</button>
            <button class="secondary-button" @click="download">Markdown</button>
            <button class="secondary-button" @click="exportReport('docx')">Word</button>
            <button class="secondary-button" @click="exportReport('pdf')">PDF</button>
            <button class="danger-text-button" @click="deleteReport">删除</button>
          </div>
        </div>

        <div v-if="activeReport" class="report-task-state">
          <h3>{{ selected?.status_text }}</h3><el-progress :percentage="selected?.progress || 0" />
          <p>报告在后台生成，离开本页面不会中断任务。</p><button class="stop-button" @click="cancelReport">停止生成</button>
        </div>
        <div v-else-if="selected?.status === 'FAILED' || selected?.status === 'CANCELLED'" class="report-error-state">
          <h3>{{ selected.status_text }}</h3><p>{{ selected.error || '任务已停止，可以重新生成。' }}</p><button class="retry-button" @click="retryReport">重新生成</button>
        </div>
        <template v-else-if="selected">
          <div v-if="editMode" class="report-edit-form"><el-input v-model="draftTitle" size="large" /><textarea v-model="draftContent" spellcheck="false" /></div>
          <pre v-else class="report-markdown">{{ selected.content_markdown }}</pre>
        </template>
        <div v-else class="empty-state">请先在左侧配置报告，并在右侧确认参与分析的资料。</div>
      </main>

      <aside class="section-card evidence-panel">
        <div class="evidence-tabs"><button :class="{active:rightTab==='materials'}" @click="rightTab='materials'">资料选择</button><button :class="{active:rightTab==='citations'}" @click="rightTab='citations'">报告引用 {{ selected?.citations?.length || 0 }}</button><button :class="{active:rightTab==='versions'}" @click="rightTab='versions'">版本 {{ versions.length }}</button></div>
        <template v-if="rightTab === 'materials'">
          <el-input v-model="materialSearch" clearable placeholder="搜索标题或来源" />
          <div class="material-actions"><span>已选 {{ selectedDocumentIds.length }} 篇</span><button @click="selectVisible">选择可用资料</button><button @click="selectedDocumentIds=[]">清空</button></div>
          <div v-loading="loadingMaterials" class="material-list">
            <label v-for="item in visibleMaterials" :key="item.id" class="material-item" :class="{unusable:!item.usable}">
              <input v-model="selectedDocumentIds" type="checkbox" :value="item.id" :disabled="!item.usable || selectedDocumentIds.length >= 50 && !selectedDocumentIds.includes(item.id)" />
              <span><strong>{{ item.title }}</strong><small>{{ SOURCE_LABELS[item.document_type] }} · {{ item.source_name }}</small><small>{{ item.published_at?.slice(0, 10) || '日期未知' }} · {{ item.usable ? '已索引' : item.status }}</small></span>
            </label>
            <div v-if="!visibleMaterials.length" class="mini-empty">没有匹配资料</div>
          </div>
        </template>
        <template v-else-if="rightTab === 'citations'">
          <div v-if="!selected?.citations?.length" class="mini-empty">当前报告暂无引用</div>
          <article v-for="citation in selected?.citations || []" :key="citation.number" class="evidence-card">
            <div><b>[{{ citation.number }}]</b><span>{{ citation.credibility }}</span></div><strong>{{ citation.title }}</strong><p>{{ citation.quote }}</p><small>{{ citation.source_name }} · {{ citation.published_at?.slice(0,10) || '日期未知' }}<template v-if="citation.page"> · 第{{ citation.page }}页</template></small><a :href="citation.source_url" target="_blank" rel="noopener">打开原始资料</a>
          </article>
        </template>
        <template v-else>
          <div v-if="!versions.length" class="mini-empty">当前报告暂无历史版本</div>
          <article v-for="version in versions" :key="version.id" class="version-card">
            <div><strong>V{{ version.version_number }}</strong><span>{{ version.change_type === 'GENERATED' ? '系统生成' : '人工编辑' }}</span></div>
            <small>{{ version.created_at.slice(0, 16).replace('T', ' ') }}</small>
            <button @click="restoreVersion(version)">载入此版本</button>
          </article>
        </template>
      </aside>
    </div>
  </div>
</template>
