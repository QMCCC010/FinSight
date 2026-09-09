<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import type { Company } from '../types'

type ComparisonMode = 'SELECTED' | 'CONSENSUS'

const companies = ref<Company[]>([])
const stockCode = ref('002594')
const reports = ref<any[]>([])
const selectedIds = ref<number[]>([])
const comparisonMode = ref<ComparisonMode>('SELECTED')
const selectedInstitutions = ref<string[]>([])
const normalizedRatings = ref<string[]>([])
const dateRange = ref<string[]>([])
const latestPerInstitution = ref(true)
const result = ref<any>(null)
const loadingReports = ref(false)
const loading = ref(false)
const resultPage = ref(1)
const resultPageSize = 10

const selectedCount = computed(() => selectedIds.value.length)
const institutionOptions = computed(() => [...new Set(reports.value.map(item => item.source_name).filter(Boolean))].sort())
const filteredReports = computed(() => reports.value.filter(report => {
  if (selectedInstitutions.value.length && !selectedInstitutions.value.includes(report.source_name)) return false
  const date = report.published_at?.slice(0, 10)
  if (dateRange.value?.length === 2 && date) return date >= dateRange.value[0] && date <= dateRange.value[1]
  return true
}))
const pagedResultReports = computed(() => {
  const start = (resultPage.value - 1) * resultPageSize
  return (result.value?.reports || []).slice(start, start + resultPageSize)
})

const ratingOptions = [
  { value: 'POSITIVE', label: '积极' },
  { value: 'SLIGHTLY_POSITIVE', label: '偏积极' },
  { value: 'NEUTRAL', label: '中性' },
  { value: 'SLIGHTLY_NEGATIVE', label: '偏谨慎' },
  { value: 'NEGATIVE', label: '谨慎' },
]

function isExternalSource(url?: string) {
  return Boolean(url && /^https?:\/\//i.test(url))
}

function defaultSelection(items: any[]) {
  const selected: number[] = []
  const institutions = new Set<string>()
  for (const report of items) {
    const institution = report.source_name || `document-${report.id}`
    if (institutions.has(institution)) continue
    institutions.add(institution)
    selected.push(report.id)
    if (selected.length >= 5) break
  }
  for (const report of items) {
    if (selected.length >= 5) break
    if (!selected.includes(report.id)) selected.push(report.id)
  }
  return selected
}

async function loadReports() {
  if (!stockCode.value) return
  loadingReports.value = true
  result.value = null
  selectedInstitutions.value = []
  normalizedRatings.value = []
  dateRange.value = []
  try {
    reports.value = (await api.get('/documents', { params: { stock_code: stockCode.value, document_type: 'RESEARCH_REPORT', status: 'INDEXED', limit: 200 } })).data
    selectedIds.value = defaultSelection(reports.value)
  } finally {
    loadingReports.value = false
  }
}

function selectVisibleLatest() {
  selectedIds.value = defaultSelection(filteredReports.value)
  result.value = null
}

function clearSelection() {
  selectedIds.value = []
  result.value = null
}

function clearFilters() {
  selectedInstitutions.value = []
  normalizedRatings.value = []
  dateRange.value = []
  result.value = null
}

function dateValue(value: string, end = false) {
  if (!value) return undefined
  return `${value}T${end ? '23:59:59' : '00:00:00'}`
}

async function compare() {
  if (comparisonMode.value === 'SELECTED' && (selectedCount.value < 2 || selectedCount.value > 10)) {
    ElMessage.warning('请选择2～10篇研报进行精选对比')
    return
  }
  loading.value = true
  try {
    result.value = (await api.post('/analysis/report-comparison', {
      stock_code: stockCode.value,
      comparison_mode: comparisonMode.value,
      document_ids: comparisonMode.value === 'SELECTED' ? selectedIds.value : undefined,
      limit: comparisonMode.value === 'CONSENSUS' ? 50 : selectedIds.value.length,
      institutions: comparisonMode.value === 'CONSENSUS' ? selectedInstitutions.value : [],
      normalized_ratings: comparisonMode.value === 'CONSENSUS' ? normalizedRatings.value : [],
      date_from: comparisonMode.value === 'CONSENSUS' ? dateValue(dateRange.value?.[0]) : undefined,
      date_to: comparisonMode.value === 'CONSENSUS' ? dateValue(dateRange.value?.[1], true) : undefined,
      latest_per_institution: comparisonMode.value === 'CONSENSUS' ? latestPerInstitution.value : false,
    })).data
    resultPage.value = 1
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '研报对比失败')
  } finally {
    loading.value = false
  }
}

function rangeText(range: any, unit: string) {
  if (!range) return '可比数据不足'
  return `${range.min.toFixed(2)}～${range.max.toFixed(2)}${unit}`
}

function rangeInstitutions(range: any) {
  if (!range) return ''
  return range.min_institution === range.max_institution
    ? range.min_institution
    : `${range.min_institution} → ${range.max_institution}`
}

onMounted(async () => {
  companies.value = (await api.get('/companies?tracked_only=true')).data
  if (!companies.value.some(item => item.stock_code === stockCode.value) && companies.value.length) stockCode.value = companies.value[0].stock_code
  await loadReports()
})
watch(stockCode, loadReports)
watch([selectedInstitutions, dateRange], () => {
  result.value = null
  if (comparisonMode.value === 'SELECTED') {
    const visibleIds = new Set(filteredReports.value.map(item => item.id))
    selectedIds.value = selectedIds.value.filter(id => visibleIds.has(id))
  }
}, { deep: true })
watch([comparisonMode, normalizedRatings, latestPerInstitution], () => { result.value = null }, { deep: true })
</script>

<template>
  <div class="page-wrap">
    <div class="page-heading"><div><span class="eyebrow">VIEWPOINT COMPARISON</span><h1>多研报观点对比</h1><p>精选少量研报逐篇比较，或从更大样本中提炼跨机构共识与分歧。</p></div></div>

    <section class="section-card mode-panel">
      <div class="mode-switch">
        <button :class="{ active: comparisonMode === 'SELECTED' }" @click="comparisonMode = 'SELECTED'"><strong>精选对比</strong><span>手动选择2～10篇，适合逐篇核对</span></button>
        <button :class="{ active: comparisonMode === 'CONSENSUS' }" @click="comparisonMode = 'CONSENSUS'"><strong>全量共识</strong><span>按条件分析最多50篇，默认按机构去重</span></button>
      </div>
    </section>

    <section class="filter-bar compare-filter">
      <el-select v-model="stockCode" placeholder="选择公司"><el-option v-for="item in companies" :key="item.id" :label="`${item.name} ${item.stock_code}`" :value="item.stock_code" /></el-select>
      <el-select v-model="selectedInstitutions" multiple collapse-tags clearable placeholder="筛选机构"><el-option v-for="item in institutionOptions" :key="item" :label="item" :value="item" /></el-select>
      <el-date-picker v-model="dateRange" type="daterange" value-format="YYYY-MM-DD" start-placeholder="开始日期" end-placeholder="结束日期" range-separator="至" />
      <el-select v-if="comparisonMode === 'CONSENSUS'" v-model="normalizedRatings" multiple collapse-tags clearable placeholder="评级方向"><el-option v-for="item in ratingOptions" :key="item.value" :label="item.label" :value="item.value" /></el-select>
      <button class="table-action" @click="clearFilters">清除筛选</button>
    </section>

    <section v-if="comparisonMode === 'SELECTED'" class="section-card report-picker" v-loading="loadingReports">
      <div class="section-title"><div><h2>选择研报</h2><p>默认选择来自不同机构的最新5篇；最多可扩展到10篇。</p></div><span class="selection-count" :class="{ invalid: selectedCount < 2 }">已选 {{ selectedCount }}/10 篇</span></div>
      <div class="picker-actions"><span>当前筛选显示 {{ filteredReports.length }} / {{ reports.length }} 篇</span><button @click="selectVisibleLatest">选择可见范围最新5篇</button><button @click="clearSelection">清空选择</button></div>
      <div v-if="!filteredReports.length && !loadingReports" class="empty-state">当前筛选条件下没有已完成解析的研报。</div>
      <el-checkbox-group v-else v-model="selectedIds" class="report-options" @change="result = null">
        <el-checkbox v-for="report in filteredReports" :key="report.id" :value="report.id" :disabled="selectedCount >= 10 && !selectedIds.includes(report.id)" class="report-option">
          <span class="report-option-main"><strong>{{ report.title }}</strong><small>{{ report.source_name }} · {{ report.published_at?.slice(0, 10) || '时间未知' }}</small></span>
        </el-checkbox>
      </el-checkbox-group>
    </section>

    <section v-else class="section-card consensus-scope">
      <div><h2>全量分析范围</h2><p>系统会在当前公司、机构、日期和评级条件内读取最多50篇研报，再计算跨机构共识、观点分歧、共同风险和盈利预测区间。</p></div>
      <el-switch v-model="latestPerInstitution" active-text="每家机构仅保留最新一篇" inactive-text="保留同机构多期研报" />
    </section>

    <section class="compare-submit">
      <span v-if="comparisonMode === 'SELECTED'">至少2篇才能判断共识；达到10篇后需先取消一篇再改选。</span>
      <span v-else>符合条件不足2篇时，系统会明确返回资料不足，不生成虚假共识。</span>
      <button class="primary-button inline" :disabled="loading || (comparisonMode === 'SELECTED' && selectedCount < 2)" @click="compare">{{ loading ? '正在分析…' : comparisonMode === 'SELECTED' ? '生成精选对比' : '生成全量共识' }}</button>
    </section>

    <section v-if="result" class="section-card result-panel">
      <div class="section-title"><div><h2>{{ result.company }}</h2><p>{{ result.summary || result.message }}</p></div><span class="pill">{{ result.comparison_mode === 'CONSENSUS' ? '全量共识' : '精选对比' }} · {{ result.status }}</span></div>

      <el-table :data="pagedResultReports" stripe>
        <el-table-column prop="institution" label="机构" min-width="140" />
        <el-table-column label="研报" min-width="300"><template #default="scope"><a v-if="isExternalSource(scope.row.source_url)" :href="scope.row.source_url" target="_blank" rel="noopener" class="report-link">{{ scope.row.title }}</a><span v-else>{{ scope.row.title }}</span></template></el-table-column>
        <el-table-column prop="rating" label="有效评级" width="110"><template #default="scope">{{ scope.row.rating || '未提取' }}</template></el-table-column>
        <el-table-column prop="target_price" label="目标价" width="100"><template #default="scope">{{ scope.row.target_price ?? '-' }}</template></el-table-column>
        <el-table-column prop="published_at" label="日期" width="120"><template #default="scope">{{ scope.row.published_at?.slice(0, 10) || '-' }}</template></el-table-column>
      </el-table>
      <el-pagination v-if="result.reports?.length > resultPageSize" v-model:current-page="resultPage" class="result-pagination" :page-size="resultPageSize" :total="result.reports.length" layout="prev, pager, next, total" />

      <div v-if="result.forecast_ranges?.length" class="forecast-section">
        <h3>盈利预测区间</h3>
        <p>金额统一为亿元，区间按同年度不同机构的最低值与最高值展示。</p>
        <el-table :data="result.forecast_ranges" stripe>
          <el-table-column prop="year" label="年度" width="90"><template #default="scope">{{ scope.row.year }}E</template></el-table-column>
          <el-table-column label="营业收入"><template #default="scope"><strong>{{ rangeText(scope.row.revenue, '亿元') }}</strong><small>{{ rangeInstitutions(scope.row.revenue) }}</small></template></el-table-column>
          <el-table-column label="归母净利润"><template #default="scope"><strong>{{ rangeText(scope.row.net_profit, '亿元') }}</strong><small>{{ rangeInstitutions(scope.row.net_profit) }}</small></template></el-table-column>
          <el-table-column label="EPS"><template #default="scope"><strong>{{ rangeText(scope.row.eps, '元') }}</strong><small>{{ rangeInstitutions(scope.row.eps) }}</small></template></el-table-column>
          <el-table-column prop="institution_count" label="机构数" width="90" />
        </el-table>
      </div>

      <div class="compare-summary three-columns">
        <div><h3>机构共识</h3><ul v-if="result.consensus?.length"><li v-for="item in result.consensus" :key="item">{{ item }}</li></ul><p v-else class="analysis-empty">{{ result.consensus_message }}</p></div>
        <div><h3>观点分歧</h3><ul v-if="result.differences?.length"><li v-for="item in result.differences" :key="item">{{ item }}</li></ul><p v-else class="analysis-empty">暂未抽取到明确的观点分歧。</p></div>
        <div><h3>共同风险</h3><ul v-if="result.common_risks?.length"><li v-for="item in result.common_risks" :key="item">{{ item }}</li></ul><p v-else class="analysis-empty">{{ result.common_risks_message }}</p></div>
      </div>
    </section>
    <div v-else class="empty-panel">设置分析范围并生成对比；筛选或选择变化后不会沿用上一次结果。</div>
  </div>
</template>

<style scoped>
.mode-panel { padding: 8px; }
.mode-switch { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.mode-switch button { display: grid; gap: 4px; padding: 14px 16px; text-align: left; border: 1px solid transparent; border-radius: 10px; background: transparent; color: #65736c; }
.mode-switch button.active { border-color: #a7be68; background: #f2f7e7; color: #173a2f; }
.mode-switch strong { font-size: 14px; }.mode-switch span { font-size: 11px; }
.compare-filter { display: grid; grid-template-columns: 180px minmax(180px, 1fr) minmax(300px, 1.4fr) minmax(170px, 1fr) auto; align-items: center; }
.selection-count { color: #54712e; font-size: 13px; }.selection-count.invalid { color: #a64035; }
button:disabled { opacity: .55; cursor: not-allowed; transform: none; }
.picker-actions { display: flex; align-items: center; gap: 12px; margin: 0 0 14px; color: #7d8982; font-size: 11px; }
.picker-actions span { margin-right: auto; }.picker-actions button { border: 0; background: transparent; color: #56752a; }
.report-options { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.report-option { width: 100%; height: auto; margin: 0; padding: 13px; border: 1px solid #e0e6e0; border-radius: 10px; align-items: flex-start; }
.report-option :deep(.el-checkbox__label) { min-width: 0; white-space: normal; }.report-option-main { display: block; }
.report-option-main strong,.report-option-main small { display: block; }.report-option-main small { margin-top: 5px; color: #7d8982; }
.consensus-scope { display: flex; align-items: center; justify-content: space-between; gap: 24px; }.consensus-scope h2 { margin: 0 0 7px; }.consensus-scope p { margin: 0; color: #75827b; }
.compare-submit { display: flex; align-items: center; justify-content: flex-end; gap: 18px; margin: 16px 0; }.compare-submit span { color: #7d8982; font-size: 12px; }
.report-link { color: #42641f; text-decoration: none; }.report-link:hover { text-decoration: underline; }
.result-pagination { justify-content: flex-end; margin-top: 14px; }
.forecast-section { margin-top: 28px; }.forecast-section>h3 { margin-bottom: 5px; }.forecast-section>p { color: #7e8983; font-size: 12px; }
.forecast-section strong,.forecast-section small { display: block; }.forecast-section small { margin-top: 3px; color: #859089; font-size: 10px; }
.three-columns { grid-template-columns: repeat(3, 1fr); }.analysis-empty { color: #7e8983; line-height: 1.6; }
@media (max-width: 1450px) { .compare-filter { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 1200px) { .report-options,.mode-switch,.three-columns { grid-template-columns: 1fr; }.consensus-scope { align-items: flex-start; flex-direction: column; } }
</style>
