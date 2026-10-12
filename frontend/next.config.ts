import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A self-contained server in .next/standalone, which the Docker image runs
  // with `node server.js`. `npm run dev` is unaffected.
  output: "standalone",
};

export default nextConfig;
