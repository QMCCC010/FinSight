<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api'
import type { Company } from '../types'

const companies = ref<Company[]>([])
const summary = ref<any>({})
const loading = ref(true)
const router = useRouter()
const tracked = computed(() => companies.value.filter(item => item.tracking_mode !== 'INACTIVE'))

onMounted(async () => {
  try {
    const [companyResponse, summaryResponse] = await Promise.all([api.get('/companies'), api.get('/companies/dashboard/summary')])
    companies.value = companyResponse.data
    summary.value = summaryResponse.data
  } finally { loading.value = false }
})
</script>

<template>
  <div class="page-wrap">
    <div class="page-heading"><div><span class="eyebrow">RESEARCH OVERVIEW</span><h1>研究总览</h1><p>从系统共享知识库进入公司研究与证据分析。</p></div><router-link class="primary-button inline" to="/chat">开始提问</router-link></div>
    <div class="stats-grid">
      <article class="stat-card"><span>跟踪公司</span><strong>{{ summary.tracked_companies ?? tracked.length }}</strong><small>预置、长期及最近查询</small></article>
      <article class="stat-card"><span>知识库文档</span><strong>{{ summary.documents ?? 0 }}</strong><small>已索引 {{ summary.indexed_documents ?? 0 }} 篇</small></article>
      <article class="stat-card"><span>在线采集资料</span><strong>{{ summary.live_documents ?? 0 }}</strong><small>另有 {{ summary.snapshot_documents ?? 0 }} 篇演示快照</small></article>
      <article class="stat-card accent"><span>运行中任务</span><strong>{{ summary.active_runs ?? 0 }}</strong><small>最新资料 {{ summary.latest_document_at?.slice(0,10) || '暂无' }}</small></article>
    </div>
    <section class="section-card">
      <div class="section-title"><div><h2>重点研究公司</h2><p>点击进入公司信息、研报观点与事件时间线。</p></div><span class="pill">A股 · 新能源</span></div>
      <div v-if="loading" class="empty-state">正在加载公司列表…</div>
      <div v-else class="company-grid">
        <article v-for="company in tracked" :key="company.id" class="company-card" @click="router.push(`/companies/${company.stock_code}`)">
          <div class="company-top"><span class="code">{{ company.stock_code }}</span><span class="track-tag" :class="company.tracking_mode.toLowerCase()">{{ company.tracking_mode }}</span></div>
          <h3>{{ company.name }}</h3><p>{{ company.industry || '行业待补充' }}</p>
          <div class="company-footer"><span>{{ company.exchange }}</span><span>进入研究 →</span></div>
        </article>
      </div>
    </section>
    <section class="notice-strip"><strong>证据优先</strong><span>系统会区分公司正式披露、机构预测、新闻信息和社交讨论，并为重要结论附上来源。</span></section>
  </div>
</template>
