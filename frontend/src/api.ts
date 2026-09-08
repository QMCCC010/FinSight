import axios from 'axios'

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
  timeout: 90000,
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(undefined, (error) => {
  if (error.response?.status === 401) {
    localStorage.removeItem('access_token')
    localStorage.removeItem('user')
    if (location.pathname !== '/login') location.href = '/login'
  }
  return Promise.reject(error)
})

