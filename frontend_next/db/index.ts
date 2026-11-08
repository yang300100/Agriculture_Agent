import { env } from "cloudflare:workers";
import { drizzle } from "drizzle-orm/d1";
import * as schema from "./schema";

export function getDb() {
  const bindings = env as typeof env & { DB?: D1Database };
  if (!bindings.DB) {
    throw new Error(
      "Cloudflare D1 binding `DB` is unavailable. Configure a real `DB` binding in wrangler.jsonc before using the optional D1 example.",
    );
  }

  return drizzle(bindings.DB, { schema });
}
