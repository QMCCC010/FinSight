<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const auth = useAuthStore()
const form = reactive({ username: 'admin', password: 'Admin123!' })
const loading = ref(false)
const error = ref('')

async function submit() {
  loading.value = true
  error.value = ''
  try {
    await auth.login(form.username, form.password)
    router.push('/')
  } catch (e: any) {
    error.value = e.response?.data?.detail || '登录失败，请检查后端服务。'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <section class="login-hero">
      <div class="hero-label">FINANCIAL INTELLIGENCE</div>
      <h1>让每一条研判<br />都有据可循</h1>
      <p>汇聚研报、公告、新闻与市场讨论，用RAG和Agent构建可追溯的研究视角。</p>
      <div class="hero-flow"><span>多源采集</span><i>→</i><span>结构化抽取</span><i>→</i><span>证据化研判</span></div>
    </section>
    <section class="login-panel">
      <form class="login-card" @submit.prevent="submit">
        <div class="brand brand-dark"><span class="brand-mark">F</span><div><strong>FinSight</strong><small>金融研报智能分析系统</small></div></div>
        <h2>欢迎回来</h2>
        <p>登录后访问共享金融知识库</p>
        <label>用户名<input v-model="form.username" autocomplete="username" /></label>
        <label>密码<input v-model="form.password" type="password" autocomplete="current-password" /></label>
        <div v-if="error" class="error-box">{{ error }}</div>
        <button class="primary-button" :disabled="loading">{{ loading ? '正在验证…' : '登录系统' }}</button>
        <small class="demo-tip">演示管理员：admin / Admin123!</small>
      </form>
    </section>
  </div>
</template>

