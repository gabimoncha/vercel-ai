import type {
  SharedV3Warning,
  TranscriptionModelV3,
  TranscriptionModelV3CallOptions,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  convertBase64ToUint8Array,
  createJsonResponseHandler,
  mediaTypeToExtension,
  parseProviderOptions,
  postFormDataToApi,
  type FetchFunction,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import { xaiFailedResponseHandler } from './xai-error';
import {
  xaiTranscriptionModelOptions,
  type XaiTranscriptionModelOptions,
} from './xai-transcription-options';

export type XaiTranscriptionCallOptions = Omit<
  TranscriptionModelV3CallOptions,
  'providerOptions'
> & {
  providerOptions?: {
    xai?: XaiTranscriptionModelOptions;
  };
};

interface XaiTranscriptionModelConfig {
  provider: string;
  baseURL: string | undefined;
  headers: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
  _internal?: {
    currentDate?: () => Date;
  };
}

const languageMap: Record<string, string> = {
  arabic: 'ar',
  czech: 'cs',
  danish: 'da',
  dutch: 'nl',
  english: 'en',
  filipino: 'fil',
  french: 'fr',
  german: 'de',
  hindi: 'hi',
  indonesian: 'id',
  italian: 'it',
  japanese: 'ja',
  korean: 'ko',
  macedonian: 'mk',
  malay: 'ms',
  persian: 'fa',
  polish: 'pl',
  portuguese: 'pt',
  romanian: 'ro',
  russian: 'ru',
  spanish: 'es',
  swedish: 'sv',
  thai: 'th',
  turkish: 'tr',
  vietnamese: 'vi',
};

export class XaiTranscriptionModel implements TranscriptionModelV3 {
  readonly specificationVersion = 'v3';

  get provider(): string {
    return this.config.provider;
  }

  constructor(
    readonly modelId: 'default',
    private readonly config: XaiTranscriptionModelConfig,
  ) {}

  private async getArgs({
    audio,
    mediaType,
    providerOptions,
  }: XaiTranscriptionCallOptions) {
    const warnings: SharedV3Warning[] = [];

    const xaiOptions = await parseProviderOptions({
      provider: 'xai',
      providerOptions,
      schema: xaiTranscriptionModelOptions,
    });

    const formData = new FormData();

    if (xaiOptions) {
      const transcriptionModelOptions = {
        audio_format: xaiOptions.audioFormat,
        sample_rate: xaiOptions.sampleRate,
        language: xaiOptions.language,
        format: xaiOptions.format,
        multichannel: xaiOptions.multichannel,
        channels: xaiOptions.channels,
        diarize: xaiOptions.diarize,
      };

      for (const [key, value] of Object.entries(transcriptionModelOptions)) {
        if (value != null) {
          formData.append(key, String(value));
        }
      }
    }

    const blob =
      audio instanceof Uint8Array
        ? new Blob([audio])
        : new Blob([convertBase64ToUint8Array(audio)]);
    const fileExtension = mediaTypeToExtension(mediaType);

    formData.append(
      'file',
      new File([blob], 'audio', { type: mediaType }),
      `audio.${fileExtension}`,
    );

    return {
      formData,
      warnings,
    };
  }

  async doGenerate(
    options: XaiTranscriptionCallOptions,
  ): Promise<Awaited<ReturnType<TranscriptionModelV3['doGenerate']>>> {
    const currentDate = this.config._internal?.currentDate?.() ?? new Date();
    const { formData, warnings } = await this.getArgs(options);
    const baseURL = this.config.baseURL ?? 'https://api.x.ai/v1';

    const {
      value: response,
      responseHeaders,
      rawValue: rawResponse,
    } = await postFormDataToApi({
      url: `${baseURL}/stt`,
      headers: combineHeaders(this.config.headers(), options.headers),
      formData,
      failedResponseHandler: xaiFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        xaiTranscriptionResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const segments = getSegments(response);

    return {
      text: response.text,
      segments,
      language: normalizeLanguage(response.language),
      durationInSeconds: response.duration ?? undefined,
      warnings,
      response: {
        timestamp: currentDate,
        modelId: this.modelId,
        headers: responseHeaders,
        body: rawResponse,
      },
      providerMetadata: {
        xai: {
          words: response.words,
          channels: response.channels,
        },
      },
    };
  }
}

function getSegments(response: XaiTranscriptionResponse) {
  const words =
    response.words ??
    response.channels
      ?.flatMap(channel =>
        channel.words.map(word => ({
          ...word,
          channelIndex: channel.index,
        })),
      )
      .sort((a, b) => a.start - b.start);

  return (
    words?.map(word => ({
      text: word.text,
      startSecond: word.start,
      endSecond: word.end,
    })) ?? []
  );
}

function normalizeLanguage(language: string | null | undefined) {
  const trimmedLanguage = language?.trim();

  if (!trimmedLanguage) {
    return undefined;
  }

  if (/^[a-z]{2,3}(?:-[a-z0-9]+)*$/i.test(trimmedLanguage)) {
    return trimmedLanguage.toLowerCase();
  }

  return languageMap[trimmedLanguage.toLowerCase()] ?? undefined;
}

const xaiWordSchema = z.object({
  text: z.string(),
  start: z.number(),
  end: z.number(),
  speaker: z.number().int().nullish(),
});

const xaiTranscriptionResponseSchema = z.object({
  text: z.string(),
  language: z.string().nullish(),
  duration: z.number().nullish(),
  words: z.array(xaiWordSchema).nullish(),
  channels: z
    .array(
      z.object({
        index: z.number().int(),
        text: z.string(),
        words: z.array(xaiWordSchema),
      }),
    )
    .nullish(),
});

type XaiTranscriptionResponse = z.infer<typeof xaiTranscriptionResponseSchema>;
