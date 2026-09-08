import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from './stores/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: () => import('./views/LoginView.vue'), meta: { public: true } },
    { path: '/', name: 'dashboard', component: () => import('./views/DashboardView.vue') },
    { path: '/companies/:code', name: 'company', component: () => import('./views/CompanyView.vue') },
    { path: '/compare', name: 'compare', component: () => import('./views/CompareView.vue') },
    { path: '/chat', name: 'chat', component: () => import('./views/ChatView.vue') },
    { path: '/workbench', name: 'workbench', component: () => import('./views/WorkbenchView.vue') },
    { path: '/admin', name: 'admin', component: () => import('./views/AdminView.vue'), meta: { admin: true } },
  ],
})

router.beforeEach((to) => {
  const auth = useAuthStore()
  if (!to.meta.public && !auth.token) return '/login'
  if (to.meta.admin && auth.user?.role !== 'ADMIN') return '/'
  if (to.path === '/login' && auth.token) return '/'
})

export default router
