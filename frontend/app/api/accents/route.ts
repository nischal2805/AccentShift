/**
 * GET /api/accents
 * Forwards to the FastAPI backend and returns the list of supported accents.
 */
import { NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${BACKEND}/accents`, { cache: "no-store" });
    if (!res.ok) {
      return NextResponse.json(
        { error: "Backend unavailable" },
        { status: res.status }
      );
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      {
        accents: [
          { key: "indian_english", label: "Indian English" },
          { key: "chinese_english", label: "Chinese-accented English" },
          { key: "japanese_english", label: "Japanese-accented English" },
          { key: "british_english", label: "British English" },
          { key: "american_english", label: "American English" },
        ],
      },
      { status: 200 }
    );
  }
}
