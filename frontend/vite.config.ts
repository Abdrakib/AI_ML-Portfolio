import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/predict': {
        target: 'https://ai-ml-portfolio-2pio.onrender.com',
        changeOrigin: true,
      },
      '/predict_overlay': {
        target: 'https://ai-ml-portfolio-2pio.onrender.com',
        changeOrigin: true,
      },
    },
  },
})
