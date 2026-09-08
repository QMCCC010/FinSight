import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    // Docker Desktop on Windows may update bind-mounted files without
    // forwarding a filesystem event. Polling keeps HMR and served CSS in sync.
    watch: { usePolling: true, interval: 500 },
  },
})
