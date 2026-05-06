import { z } from 'zod/v4';

export const xaiTranscriptionModelOptions = z.object({
  audioFormat: z.enum(['pcm', 'mulaw', 'alaw']).optional(),
  sampleRate: z
    .union([
      z.literal(8000),
      z.literal(16000),
      z.literal(22050),
      z.literal(24000),
      z.literal(44100),
      z.literal(48000),
    ])
    .optional(),
  language: z.string().optional(),
  format: z.boolean().optional(),
  multichannel: z.boolean().optional(),
  channels: z.number().int().min(2).max(8).optional(),
  diarize: z.boolean().optional(),
});

export type XaiTranscriptionModelOptions = z.infer<
  typeof xaiTranscriptionModelOptions
>;
