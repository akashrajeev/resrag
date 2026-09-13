import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    {
      name: 'resrag-production-api-default',
      transform(code, id) {
        if (id.endsWith('/src/main.jsx') && mode === 'production') {
          return code
            .replace("import.meta.env.VITE_API_URL || 'http://localhost:8000'", "import.meta.env.VITE_API_URL || 'https://resrag-api.onrender.com'")
            .replace('document.documentElement.dataset.theme', 'window.document.documentElement.dataset.theme');
        }
        return null;
      },
    },
  ],
  server: {
    port: 5173,
  },
  preview: {
    port: 5173,
  },
}));
