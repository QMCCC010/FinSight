<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import type { Company } from '../types'

const companies = ref<Company[]>([])
const stockCode = ref('002594')
const reports = ref<any[]>([])
const selectedIds = ref<number[]>([])
const result = ref<any>(null)
const loadingReports = ref(false)
const loading = ref(false)
const selectedCount = computed(() => selectedIds.value.length)

function isExternalSource(url?: string) {
  return Boolean(url && /^https?:\/\//i.test(url))
}

async function loadReports() {
  if (!stockCode.value) return
  loadingReports.value = true
  result.value = null
  try {
    reports.value = (await api.get('/documents', { params: { stock_code: stockCode.value, document_type: 'RESEARCH_REPORT', status: 'INDEXED', limit: 50 } })).data
    selectedIds.value = reports.value.slice(0, 5).map(item => item.id)
  } finally {
    loadingReports.value = false
  }
}

async function compare() {
  if (selectedIds.value.length < 2 || selectedIds.value.length > 5) {
    ElMessage.warning('请选择2～5篇研报进行对比')
    return
  }
  loading.value = true
  try {
    result.value = (await api.post('/analysis/report-comparison', {
      stock_code: stockCode.value,
      document_ids: selectedIds.value,
      limit: selectedIds.value.length,
    })).data
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  companies.value = (await api.get('/companies?tracked_only=true')).data
  if (!companies.value.some(item => item.stock_code === stockCode.value) && companies.value.length) stockCode.value = companies.value[0].stock_code
  await loadReports()
})
watch(stockCode, loadReports)
</script>

<template>
  <div class="page-wrap">
    <div class="page-heading"><div><span class="eyebrow">VIEWPOINT COMPARISON</span><h1>多研报观点对比</h1><p>选择2～5篇研报，比较评级、预测、共识、分歧和共同风险。</p></div></div>
    <section class="filter-bar compare-filter">
      <el-select v-model="stockCode" placeholder="选择公司"><el-option v-for="item in companies" :key="item.id" :label="`${item.name} ${item.stock_code}`" :value="item.stock_code" /></el-select>
      <span class="selection-count" :class="{ invalid: selectedCount < 2 }">已选 {{ selectedCount }}/5 篇</span>
      <button class="primary-button inline" :disabled="loading || selectedCount < 2" @click="compare">{{ loading ? '正在比较…' : '生成对比' }}</button>
    </section>

    <section class="section-card report-picker" v-loading="loadingReports">
      <div class="section-title"><div><h2>选择研报</h2><p>默认选中最新5篇；达到5篇后需先取消一篇才能改选。</p></div><span class="pill">共 {{ reports.length }} 篇</span></div>
      <div v-if="!reports.length && !loadingReports" class="empty-state">该公司还没有已完成解析的研报，请先在知识库管理后台同步。</div>
      <el-checkbox-group v-else v-model="selectedIds" class="report-options" @change="result = null">
        <el-checkbox v-for="report in reports" :key="report.id" :value="report.id" :disabled="selectedCount >= 5 && !selectedIds.includes(report.id)" class="report-option">
          <span class="report-option-main"><strong>{{ report.title }}</strong><small>{{ report.source_name }} · {{ report.published_at?.slice(0, 10) || '时间未知' }}</small></span>
        </el-checkbox>
      </el-checkbox-group>
    </section>

    <section v-if="result" class="section-card">
      <div class="section-title"><div><h2>{{ result.company }}</h2><p>{{ result.summary || result.message }}</p></div><span class="pill">{{ result.status }}</span></div>
      <el-table :data="result.reports" stripe>
        <el-table-column prop="institution" label="机构" min-width="140" />
        <el-table-column label="研报" min-width="280"><template #default="scope"><a v-if="isExternalSource(scope.row.source_url)" :href="scope.row.source_url" target="_blank" rel="noopener" class="report-link">{{ scope.row.title }}</a><span v-else>{{ scope.row.title }}</span></template></el-table-column>
        <el-table-column prop="rating" label="有效评级" width="110"><template #default="scope">{{ scope.row.rating || '未提取' }}</template></el-table-column>
        <el-table-column prop="target_price" label="目标价" width="100"><template #default="scope">{{ scope.row.target_price ?? '-' }}</template></el-table-column>
        <el-table-column prop="published_at" label="日期" width="120"><template #default="scope">{{ scope.row.published_at?.slice(0, 10) || '-' }}</template></el-table-column>
      </el-table>
      <div class="compare-summary three-columns">
        <div><h3>机构共识</h3><ul v-if="result.consensus?.length"><li v-for="item in result.consensus" :key="item">{{ item }}</li></ul><p v-else class="analysis-empty">{{ result.consensus_message }}</p></div>
        <div><h3>观点分歧</h3><ul v-if="result.differences?.length"><li v-for="item in result.differences" :key="item">{{ item }}</li></ul><p v-else class="analysis-empty">暂未抽取到明确的观点分歧。</p></div>
        <div><h3>共同风险</h3><ul v-if="result.common_risks?.length"><li v-for="item in result.common_risks" :key="item">{{ item }}</li></ul><p v-else class="analysis-empty">{{ result.common_risks_message }}</p></div>
      </div>
    </section>
    <div v-else class="empty-panel">选择2～5篇研报后生成观点对比。结果变化前不会沿用上一次分析。</div>
  </div>
</template>

<style scoped>
.compare-filter { align-items: center; }
.selection-count { color: #54712e; font-size: 13px; }
.selection-count.invalid { color: #a64035; }
button:disabled { opacity: .55; cursor: not-allowed; transform: none; }
.report-options { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.report-option { width: 100%; height: auto; margin: 0; padding: 13px; border: 1px solid #e0e6e0; border-radius: 10px; align-items: flex-start; }
.report-option :deep(.el-checkbox__label) { min-width: 0; white-space: normal; }
.report-option-main { display: block; }
.report-option-main strong, .report-option-main small { display: block; }
.report-option-main small { margin-top: 5px; color: #7d8982; }
.report-link { color: #42641f; text-decoration: none; }
.report-link:hover { text-decoration: underline; }
.three-columns { grid-template-columns: repeat(3, 1fr); }
.analysis-empty { color: #7e8983; line-height: 1.6; }
@media (max-width: 1200px) { .report-options { grid-template-columns: 1fr; } .three-columns { grid-template-columns: 1fr; } }
</style>
