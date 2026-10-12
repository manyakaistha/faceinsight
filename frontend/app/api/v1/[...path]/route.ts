import { NextRequest } from "next/server";
export const dynamic = "force-dynamic";
async function proxy(
  request: NextRequest,
  { params }: { params: { path: string[] } },
) {
  const origin = process.env.FACEINSIGHT_API_URL || "http://127.0.0.1:8100";
  try {
    const headers = new Headers();
    const type = request.headers.get("content-type");
    if (type) headers.set("content-type", type);
    const response = await fetch(
      `${origin}/api/v1/${params.path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`,
      {
        method: request.method,
        headers,
        body:
          request.method === "POST" ? await request.arrayBuffer() : undefined,
        cache: "no-store",
        signal: AbortSignal.timeout(120000),
      },
    );
    return new Response(response.body, {
      status: response.status,
      headers: {
        "content-type":
          response.headers.get("content-type") || "application/json",
      },
    });
  } catch {
    return Response.json(
      {
        detail:
          "The model server is unavailable. Start the backend on port 8100 and retry.",
      },
      { status: 503 },
    );
  }
}
export { proxy as GET, proxy as POST };
