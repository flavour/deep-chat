/* eslint-disable no-restricted-globals */
import { pipeline } from "@huggingface/transformers";

// Message format: { audioBuffer: ArrayBuffer, config: { model, driver, pipelineOptions } }
self.onmessage = async (evt: MessageEvent) => {
  const { audioBuffer, config } = evt.data;
  try {
    // Prepare pipeline
    const transcriber = await pipeline(
      "automatic-speech-recognition",
      config.model,
      config.pipelineOptions || {}
    );
    // Run transcription
    const output = await transcriber(audioBuffer);
    // Post result back to main thread
    if (Array.isArray(output)) {
      self.postMessage({ text: output[0]?.text ?? "" });
    } else {
      self.postMessage({ text: output.text });
    }
  } catch (err: unknown) {
    let errorMsg: string;
    if (
      err &&
      typeof err === "object" &&
      "message" in err &&
      typeof (err as { message?: unknown }).message === "string"
    ) {
      errorMsg = (err as { message: string }).message;
    } else {
      errorMsg = String(err);
    }
    self.postMessage({ error: errorMsg });
  }
};