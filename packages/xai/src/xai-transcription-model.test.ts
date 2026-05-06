import { createTestServer } from '@ai-sdk/test-server/with-vitest';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { createXai } from './xai-provider';
import { XaiTranscriptionModel } from './xai-transcription-model';

vi.mock('./version', () => ({
  VERSION: '0.0.0-test',
}));

const transcriptionResponse = {
  text: 'Hello from the Vercel AI SDK.',
  language: 'English',
  duration: 1.25,
  words: [
    { text: 'Hello', start: 0, end: 0.4, speaker: 0 },
    { text: 'from', start: 0.4, end: 0.7, speaker: 0 },
    { text: 'Vercel', start: 0.7, end: 1.25, speaker: 1 },
  ],
};

const audioData = await readFile(path.join(__dirname, 'transcript-test.mp3'));

const server = createTestServer({
  'https://api.x.ai/v1/stt': {
    response: {
      type: 'json-value',
      headers: {
        'x-request-id': 'request-id',
      },
      body: transcriptionResponse,
    },
  },
});

function createModel({
  headers,
  currentDate,
  baseURL,
}: {
  headers?: () => Record<string, string | undefined>;
  currentDate?: () => Date;
  baseURL?: string;
} = {}) {
  return new XaiTranscriptionModel('default', {
    provider: 'xai.transcription',
    baseURL: baseURL ?? 'https://api.x.ai/v1',
    headers: headers ?? (() => ({ authorization: 'Bearer test-api-key' })),
    _internal: {
      currentDate,
    },
  });
}

describe('XaiTranscriptionModel', () => {
  it('should expose correct provider and model information', () => {
    const model = createModel();

    expect(model.provider).toBe('xai.transcription');
    expect(model.modelId).toBe('default');
    expect(model.specificationVersion).toBe('v3');
  });

  it('should post to the xAI STT endpoint', async () => {
    const provider = createXai({ apiKey: 'test-api-key' });

    await provider.transcription().doGenerate({
      audio: audioData,
      mediaType: 'audio/wav',
    });

    expect(server.calls[0].requestMethod).toBe('POST');
    expect(server.calls[0].requestUrl).toBe('https://api.x.ai/v1/stt');
  });

  it('should pass headers', async () => {
    const provider = createXai({
      apiKey: 'test-api-key',
      headers: {
        'Custom-Provider-Header': 'provider-header-value',
      },
    });

    await provider.transcription().doGenerate({
      audio: audioData,
      mediaType: 'audio/wav',
      headers: {
        'Custom-Request-Header': 'request-header-value',
      },
    });

    expect(server.calls[0].requestHeaders).toMatchObject({
      authorization: 'Bearer test-api-key',
      'content-type': expect.stringMatching(
        /^multipart\/form-data; boundary=----formdata-undici-\d+$/,
      ),
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
    expect(server.calls[0].requestUserAgent).toContain('ai-sdk/xai/0.0.0-test');
  });

  it('should pass provider options and append file last', async () => {
    let capturedFormData: FormData | undefined;
    const fetch = vi.fn(async (_url, init) => {
      capturedFormData = init?.body as FormData;
      return new Response(JSON.stringify(transcriptionResponse), {
        headers: { 'content-type': 'application/json' },
      });
    });
    const model = new XaiTranscriptionModel('default', {
      provider: 'xai.transcription',
      baseURL: 'https://api.x.ai/v1',
      headers: () => ({ authorization: 'Bearer test-api-key' }),
      fetch,
    });

    await model.doGenerate({
      audio: audioData,
      mediaType: 'audio/wav',
      providerOptions: {
        xai: {
          audioFormat: 'pcm',
          sampleRate: 16000,
          language: 'en',
          format: true,
          multichannel: true,
          channels: 2,
          diarize: true,
        },
      },
    });

    expect([...capturedFormData!.keys()]).toStrictEqual([
      'audio_format',
      'sample_rate',
      'language',
      'format',
      'multichannel',
      'channels',
      'diarize',
      'file',
    ]);
    expect(Object.fromEntries(capturedFormData!.entries())).toMatchObject({
      audio_format: 'pcm',
      sample_rate: '16000',
      language: 'en',
      format: 'true',
      multichannel: 'true',
      channels: '2',
      diarize: 'true',
    });
  });

  it('should send Uint8Array audio as a multipart file', async () => {
    const model = createModel();

    await model.doGenerate({
      audio: audioData,
      mediaType: 'audio/wav',
    });

    const multipart = await server.calls[0].requestBodyMultipart;
    expect(multipart!.file).toBeInstanceOf(File);
    expect(multipart!.file.type).toBe('audio/wav');
    expect(multipart!.file.name).toBe('audio.wav');
    expect(multipart!.file.size).toBe(40169);
  });

  it('should send base64 audio as a multipart file', async () => {
    const model = createModel();

    await model.doGenerate({
      audio: 'aGVsbG8=',
      mediaType: 'audio/mp4',
    });

    const multipart = await server.calls[0].requestBodyMultipart;
    expect(multipart!.file).toBeInstanceOf(File);
    expect(multipart!.file.type).toBe('audio/mp4');
    expect(multipart!.file.name).toBe('audio.m4a');
    expect(multipart!.file.size).toBe(5);
  });

  it('should map response fields', async () => {
    const testDate = new Date(0);
    const model = createModel({
      currentDate: () => testDate,
    });

    const result = await model.doGenerate({
      audio: audioData,
      mediaType: 'audio/wav',
    });

    expect(result).toMatchObject({
      text: 'Hello from the Vercel AI SDK.',
      durationInSeconds: 1.25,
      language: 'en',
      segments: [
        { text: 'Hello', startSecond: 0, endSecond: 0.4 },
        { text: 'from', startSecond: 0.4, endSecond: 0.7 },
        { text: 'Vercel', startSecond: 0.7, endSecond: 1.25 },
      ],
      warnings: [],
      response: {
        timestamp: testDate,
        modelId: 'default',
        headers: {
          'content-type': 'application/json',
          'x-request-id': 'request-id',
        },
        body: transcriptionResponse,
      },
      providerMetadata: {
        xai: {
          words: transcriptionResponse.words,
        },
      },
    });
  });

  it('should return undefined for empty language values', async () => {
    server.urls['https://api.x.ai/v1/stt'].response = {
      type: 'json-value',
      body: {
        text: 'Hello.',
        language: '',
        duration: 1,
        words: [],
      },
    };

    const result = await createModel().doGenerate({
      audio: audioData,
      mediaType: 'audio/wav',
    });

    expect(result.language).toBeUndefined();
  });

  it('should pass through ISO-like language values', async () => {
    server.urls['https://api.x.ai/v1/stt'].response = {
      type: 'json-value',
      body: {
        text: 'Bonjour.',
        language: 'fr',
        duration: 1,
        words: [],
      },
    };

    const result = await createModel().doGenerate({
      audio: audioData,
      mediaType: 'audio/wav',
    });

    expect(result.language).toBe('fr');
  });

  it('should flatten multichannel words into segments and preserve channel metadata', async () => {
    const channels = [
      {
        index: 0,
        text: 'Agent hello.',
        words: [{ text: 'Agent', start: 0.5, end: 0.8 }],
      },
      {
        index: 1,
        text: 'Customer hi.',
        words: [{ text: 'Customer', start: 0.1, end: 0.4 }],
      },
    ];

    server.urls['https://api.x.ai/v1/stt'].response = {
      type: 'json-value',
      body: {
        text: 'Customer hi. Agent hello.',
        language: 'English',
        duration: 1,
        channels,
      },
    };

    const result = await createModel().doGenerate({
      audio: audioData,
      mediaType: 'audio/wav',
    });

    expect(result.segments).toStrictEqual([
      { text: 'Customer', startSecond: 0.1, endSecond: 0.4 },
      { text: 'Agent', startSecond: 0.5, endSecond: 0.8 },
    ]);
    expect(result.providerMetadata).toStrictEqual({
      xai: {
        words: undefined,
        channels,
      },
    });
  });
});
