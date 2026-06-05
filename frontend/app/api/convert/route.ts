/**
 * POST /api/convert
 * Proxies the multipart upload to the FastAPI backend.
 * Keeps BACKEND_URL server-side only (never exposed to browser).
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.arrayBuffer();
  const contentType = req.headers.get("content-type") ?? "multipart/form-data";

  const res = await fetch(`${BACKEND}/convert`, {
    method: "POST",
    headers: { "content-type": contentType },
    body,
  });

  if (!res.ok) {
    const errorText = await res.text();
    return NextResponse.json(
      { error: errorText || "Backend error" },
      { status: res.status }
    );
  }

  const data = await res.json();
  return NextResponse.json(data);
}
