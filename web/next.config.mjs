const API = process.env.JUMPLAB_API || "http://localhost:8000";

/** Alles unter /api geht an FastAPI. Damit hat der Browser eine Herkunft und
 *  es braucht kein CORS - eine bewegliche Teil weniger. */
export default {
  output: "standalone",
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};
