import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the app calls a relative `/api`; this proxy forwards those calls to
// the FastAPI backend so the browser sees a single origin (no CORS dance).
// In production, `npm run build` emits ./dist, which FastAPI serves itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
