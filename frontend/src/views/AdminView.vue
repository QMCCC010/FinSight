<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const sources = ref<any[]>([])
const runs = ref<any[]>([])
const documents = ref<any[]>([])
const syncing = ref(false)
const indexStatus = ref<any>(null)
let timer: number | undefined
async function load() { const [a, b, c, d] = await Promise.all([api.get('/admin/sources'), api.get('/admin/crawl-runs'), api.get('/documents?limit=30'), api.get('/admin/index/status')]); sources.value = a.data; runs.value = b.data; documents.value = c.data; indexStatus.value = d.data }
async function toggle(source: any) { await api.patch(`/admin/sources/${source.id}`, { enabled: source.enabled }); ElMessage.success('数据源状态已更新') }
async function syncAll() { syncing.value = true; try { await api.post('/admin/crawl-runs', { stock_codes: [], source_types: [] }); ElMessage.success('已创建自动采集任务'); await load() } finally { syncing.value = false } }
async function rebuild() { await api.post('/admin/index/rebuild'); ElMessage.success('索引重建任务已提交') }
async function syncMarket() { const { data } = await api.post('/admin/market/sync'); ElMessage.success(`已提交 ${data.submitted} 家公司的行情同步`) }
async function reprocess(id: number) { await api.post(`/admin/documents/${id}/reprocess`); ElMessage.success('重新处理任务已提交') }
async function reprocessBatch(failedOnly = false) { const { data } = await api.post(`/admin/documents/reprocess-batch?failed_only=${failedOnly}&limit=100`); ElMessage.success(`已提交 ${data.submitted} 篇资料重新处理`) }
async function repairCninfo() { const { data } = await api.post('/admin/documents/reprocess-batch', null, { params: { document_type: 'ANNOUNCEMENT', source_name: '巨潮资讯', legacy_only: true, limit: 500 } }); ElMessage.success('已提交 ' + data.submitted + ' 篇历史巨潮公告修复') }
async function removeDocument(id: number) { await api.delete(`/admin/documents/${id}`); ElMessage.success('文档已软删除并已提交向量删除任务'); await load() }
onMounted(async () => { await load(); timer = window.setInterval(load, 4000) })
onUnmounted(() => timer && window.clearInterval(timer))
</script>

<template>
  <div class="page-wrap">
    <div class="page-heading"><div><span class="eyebrow">KNOWLEDGE OPERATIONS</span><h1>知识库管理后台</h1><p>管理系统级共享知识库的数据源、采集任务、行情和索引状态。</p></div><div class="button-group"><button class="secondary-button" @click="syncMarket">同步行情</button><button class="secondary-button" @click="rebuild">重建索引</button><button class="primary-button inline" @click="syncAll">{{ syncing ? '提交中…' : '全部立即同步' }}</button></div></div>
    <section class="section-card"><div class="section-title"><div><h2>数据源</h2><p>四类公开金融信息源，展示真实运行耗时、成功次数和连续失败次数。</p></div><span class="pill">AUTO</span></div><div class="source-grid"><article v-for="source in sources" :key="source.id"><div><span class="doc-type">{{ source.source_type }}</span><h3>{{ source.name }}</h3><p>{{ source.adapter }} · {{ source.schedule }}</p></div><el-switch v-model="source.enabled" @change="toggle(source)" /><small :class="source.last_error ? 'source-error' : ''">{{ source.last_error ? `异常：${source.last_error.slice(0,80)}` : `最近成功：${source.last_success_at?.slice(0,16).replace('T',' ') || '尚未运行'}` }}<br>耗时 {{ source.last_duration_ms ?? '-' }} ms · 成功 {{ source.total_successes }} · 失败 {{ source.total_failures }} · 连续失败 {{ source.consecutive_failures }}</small></article></div></section>
    <section class="section-card"><div class="section-title"><div><h2>RAG索引</h2><p>Milvus执行中文BM25与BGE稠密向量混合检索；连接异常时可降级到本地FAISS/BM25。</p></div><div class="button-group"><button class="secondary-button" @click="repairCninfo">修复历史巨潮公告</button><button class="secondary-button" @click="reprocessBatch(true)">重处理失败资料</button><button class="secondary-button" @click="reprocessBatch(false)">重处理最近100篇</button></div></div><div class="index-health"><span :class="indexStatus?.compatible ? 'health-ok' : 'health-warning'">{{ indexStatus?.compatible ? '索引兼容' : '需要重建' }}</span><span>后端 {{ indexStatus?.backend || '未知' }}</span><span v-if="indexStatus?.synchronized !== undefined" :class="indexStatus.synchronized ? 'health-ok' : 'health-warning'">{{ indexStatus.synchronized ? '数据已同步' : '数据同步中' }}</span><span>向量数 {{ indexStatus?.metadata?.actual_count ?? 0 }} / 切片 {{ indexStatus?.metadata?.database_count ?? '-' }}</span><span>维度 {{ indexStatus?.metadata?.actual_dimension ?? '-' }}</span><span>模型 {{ indexStatus?.metadata?.embedding_model || '未记录' }}</span><span v-if="indexStatus?.error" class="source-error">异常：{{ indexStatus.error }}</span></div></section>
    <section class="section-card"><div class="section-title"><div><h2>采集任务</h2><p>每4秒自动刷新任务阶段与处理数量。</p></div></div><el-table :data="runs" stripe><el-table-column prop="job_id" label="任务" min-width="170"><template #default="scope"><code>{{ scope.row.job_id.slice(0,8) }}</code></template></el-table-column><el-table-column prop="trigger_type" label="触发" width="110" /><el-table-column prop="status" label="状态" width="110" /><el-table-column prop="stage" label="当前阶段" min-width="180" /><el-table-column label="进度" width="170"><template #default="scope"><el-progress :percentage="scope.row.progress" /></template></el-table-column><el-table-column prop="created_count" label="新增" width="75" /><el-table-column prop="duplicate_count" label="重复" width="75" /><el-table-column prop="failed_count" label="失败" width="75" /></el-table></section>
    <section class="section-card"><div class="section-title"><div><h2>最近文档</h2><p>查看解析状态，对失败或错误资料执行维护操作。</p></div></div><el-table :data="documents" stripe><el-table-column prop="title" label="标题" min-width="300" /><el-table-column prop="document_type" label="类型" width="150" /><el-table-column prop="source_name" label="来源" width="150" /><el-table-column label="采集模式" width="110"><template #default="scope"><span :class="['source-mode', scope.row.acquisition_mode?.toLowerCase()]">{{ scope.row.acquisition_mode }}</span></template></el-table-column><el-table-column prop="status" label="状态" width="110" /><el-table-column label="操作" width="160"><template #default="scope"><button class="table-action" @click="reprocess(scope.row.id)">重处理</button><button class="table-action danger" @click="removeDocument(scope.row.id)">删除</button></template></el-table-column></el-table></section>
  </div>
</template>
