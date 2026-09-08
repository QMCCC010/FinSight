import { defineStore } from 'pinia'
import { api } from '../api'

export interface User { id: number; username: string; role: 'ADMIN' | 'ANALYST' }

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem('access_token') || '',
    user: JSON.parse(localStorage.getItem('user') || 'null') as User | null,
  }),
  actions: {
    async login(username: string, password: string) {
      const { data } = await api.post('/auth/login', { username, password })
      this.token = data.access_token
      this.user = data.user
      localStorage.setItem('access_token', data.access_token)
      localStorage.setItem('user', JSON.stringify(data.user))
    },
    logout() {
      this.token = ''
      this.user = null
      localStorage.removeItem('access_token')
      localStorage.removeItem('user')
    },
  },
})

