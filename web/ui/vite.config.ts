import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5180, proxy: { "/api": `http://127.0.0.1:${process.env.TA_WEB_PORT ?? 8765}` } },
});
