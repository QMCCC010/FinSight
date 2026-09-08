<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from './stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const isLogin = computed(() => route.name === 'login')
const isAdmin = computed(() => auth.user?.role === 'ADMIN')

function logout() {
  auth.logout()
  router.push('/login')
}
</script>

<template>
  <router-view v-if="isLogin" />
  <div v-else class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-mark">F</span>
        <div><strong>FinSight</strong><small>金融研究智能中枢</small></div>
      </div>
      <nav>
        <router-link to="/">研究总览</router-link>
        <router-link to="/compare">研报对比</router-link>
        <router-link to="/chat">智能问答</router-link>
        <router-link to="/workbench">内容工作台</router-link>
        <router-link v-if="isAdmin" to="/admin">知识库管理</router-link>
      </nav>
      <div class="sidebar-footer">
        <span>{{ auth.user?.username }}</span>
        <small>{{ auth.user?.role }}</small>
        <button class="text-button" @click="logout">退出登录</button>
      </div>
    </aside>
    <main class="main-content">
      <header class="topbar">
        <div><span class="live-dot"></span> 系统共享知识库</div>
        <div class="disclaimer">仅供研究辅助，不构成投资建议</div>
      </header>
      <router-view />
    </main>
  </div>
</template>

