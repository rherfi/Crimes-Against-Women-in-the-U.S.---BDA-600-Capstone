import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Minimal Vite config.
export default defineConfig({
  plugins: [react()],
  server: {
    // Bind IPv4 explicitly — avoids some "connection failed" cases where
    // `localhost` resolves to ::1 but the server was only on IPv4.
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
});

