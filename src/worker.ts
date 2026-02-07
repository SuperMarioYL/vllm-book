interface Env {
  ASSETS: Fetcher;
}

const BAIDU_VERIFY_HTML_PATH = "/baidu_verify_codeva-ghdExOm4bb.html";
const BAIDU_VERIFY_FILE_CONTENT = "d5f33923dba6860243862e12f743f5c1";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === BAIDU_VERIFY_HTML_PATH) {
      // Return file content directly to avoid .html -> non-.html redirect.
      return new Response(BAIDU_VERIFY_FILE_CONTENT, {
        status: 200,
        headers: {
          "content-type": "text/html; charset=utf-8",
          "cache-control": "no-store",
        },
      });
    }

    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
