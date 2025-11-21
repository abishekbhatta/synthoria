import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  /**
   * Specify your server-side environment variables schema here. This way you can ensure the app
   * isn't built with invalid env vars.
   */
  server: {
    DATABASE_URL: z.string().url(),
    NODE_ENV: z
      .enum(["development", "test", "production"])
      .default("development"),
      MODAL_KEY : z.string(),
      MODAL_SECRET: z.string(),
      AWS_ACCESS_KEY_ID : z.string(),
      AWS_SECRET_KEY_ID : z.string(),        /* The keys are confidentials, so they are stored in server instead on the client */
      AWS_REGION : z.string(),
      S3_BUCKET : z.string(),
      SIMPLE_MODE : z.string(),
      CUSTOM_MODE_AUTO_LYRIC : z.string(),
      CUSTOM_MODE_MANUAL_LYRIC : z.string(),
      BETTER_AUTH_SECRET: z.string(),
  },

  /**
   * Specify your client-side environment variables schema here. This way you can ensure the app
   * isn't built with invalid env vars. To expose them to the client, prefix them with
   * `NEXT_PUBLIC_`.
   */
  client: {
    // NEXT_PUBLIC_CLIENTVAR: z.string(),
  },

  /**
   * You can't destruct `process.env` as a regular object in the Next.js edge runtimes (e.g.
   * middlewares) or client-side so we need to destruct manually.
   */
  runtimeEnv: {
    
    DATABASE_URL: process.env.DATABASE_URL,
    NODE_ENV: process.env.NODE_ENV,
    MODAL_KEY: process.env.MODAL_KEY,
    MODAL_SECRET: process.env.MODAL_SECRET,
    AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID,
    AWS_SECRET_KEY_ID: process.env.AWS_SECRET_KEY_ID,
    AWS_REGION : process.env.AWS_REGION,
    S3_BUCKET : process.env.S3_BUCKET,
    SIMPLE_MODE : process.env.SIMPLE_MODE,
    CUSTOM_MODE_AUTO_LYRIC : process.env.CUSTOM_MODE_AUTO_LYRIC,
    CUSTOM_MODE_MANUAL_LYRIC : process.env.CUSTOM_MODE_MANUAL_LYRIC,
    BETTER_AUTH_SECRET: process.env.BETTER_AUTH_SECRET,

    // NEXT_PUBLIC_CLIENTVAR: process.env.NEXT_PUBLIC_CLIENTVAR,
  },
  /**
   * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially
   * useful for Docker builds.
   */
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  /**
   * Makes it so that empty strings are treated as undefined. `SOME_VAR: z.string()` and
   * `SOME_VAR=''` will throw an error.
   */
  emptyStringAsUndefined: true,
});
