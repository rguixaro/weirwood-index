import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const originToken = env.VITE_DEV_ORIGIN_TOKEN ?? "local-development";

  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        "/api/catalog": {
          target: env.VITE_DEV_API_ORIGIN ?? "http://127.0.0.1:8000",
          changeOrigin: true,
          rewrite: () => "/v1/catalog",
          headers: {
            "X-Weirwood-Origin-Token": originToken
          }
        },
        "/api/search": {
          target: env.VITE_DEV_API_ORIGIN ?? "http://127.0.0.1:8000",
          changeOrigin: true,
          rewrite: () => "/v1/search",
          headers: {
            "X-Weirwood-Origin-Token": originToken
          }
        }
      }
    },
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: "./src/test/setup.ts",
      include: ["src/**/*.test.{ts,tsx}", "worker/**/*.test.ts"]
    }
  };
});
