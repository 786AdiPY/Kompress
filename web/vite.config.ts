import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
// The dev server proxies '/api' -> the FastAPI backend so the client can use a
// relative base URL in development (see src/api/client.ts).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Backend routes live at the root (/health, /runs, ...), so strip the
        // '/api' prefix the client uses as its dev base URL.
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
