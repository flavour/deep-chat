/**
 * Local SpeechEvents interface (duplicated from types/microphone.ts due to import issues)
 */
interface SpeechEvents {
  onStart?: () => void;
  onStop?: () => void;
  onResult?: (text: string, isFinal: boolean) => void;
  onPreResult?: (text: string, isFinal: boolean) => void;
  onCommandModeTrigger?: (isStart: boolean) => void;
  onPauseTrigger?: (isStart: boolean) => void;
}
import {ValidationHandler} from '../../../../../../types/validationHandler';
import {SpeechToTextConfig, TransformersOptions} from '../../../../../../types/microphone';
import {OnPreResult} from 'speech-to-element/dist/types/options';
import {TextInputEl} from '../../../textInput/textInput';
import {Messages} from '../../../../messages/messages';
import {MicrophoneButton} from '../microphoneButton';
import {DeepChat} from '../../../../../../deepChat';
import SpeechToElement from 'speech-to-element';
import {SilenceSubmit} from './silenceSubmit';

export type ProcessedConfig = SpeechToTextConfig & {onPreResult?: OnPreResult};

export type AddErrorMessage = Messages['addNewErrorMessage'];

export class SpeechToText extends MicrophoneButton {
  private readonly _addErrorMessage: AddErrorMessage;
  private _silenceSubmit?: SilenceSubmit;
  private _validationHandler?: ValidationHandler;
  public static readonly MICROPHONE_RESET_TIMEOUT_MS = 300;

  constructor(deepChat: DeepChat, textInput: TextInputEl, addErrorMessage: AddErrorMessage) {
    const config = typeof deepChat.speechToText === 'object' ? deepChat.speechToText : {};
    super(config?.button);
    const {serviceName, processedConfig} = this.processConfiguration(textInput, deepChat.speechToText);
    this._addErrorMessage = addErrorMessage;
    if (serviceName === 'webspeech' && !SpeechToElement.isWebSpeechSupported()) {
      this.changeToUnsupported();
    } else {
      const isInputEnabled = !deepChat.textInput || !deepChat.textInput.disabled;
      this.elementRef.onclick = this.buttonClick.bind(this, textInput, isInputEnabled, serviceName, processedConfig);
    }
    setTimeout(() => {
      this._validationHandler = deepChat._validationHandler;
    });
  }

  // prettier-ignore
  private processConfiguration(textInput: TextInputEl, config?: boolean | SpeechToTextConfig):
      {serviceName: string, processedConfig: ProcessedConfig} {
    const newConfig = typeof config === 'object' ? config : {};
    const webSpeechConfig = typeof newConfig.webSpeech === 'object' ? newConfig.webSpeech : {};
    const azureConfig = newConfig.azure || {};
    const processedConfig: ProcessedConfig = {
      displayInterimResults: newConfig.displayInterimResults ?? undefined,
      textColor: newConfig.textColor ?? undefined,
      translations: newConfig.translations ?? undefined,
      commands: newConfig.commands ?? undefined,
      events: newConfig.events ?? undefined,
      ...webSpeechConfig,
      ...azureConfig,
    };
    const submitPhrase = newConfig.commands?.submit;
    if (submitPhrase) {
      processedConfig.onPreResult = (text: string) => {
        if (text.toLowerCase().includes(submitPhrase)) {
          // wait for command words to be removed
          setTimeout(() => textInput.submit?.());
          SpeechToElement.endCommandMode();
          return {restart: true, removeNewText: true};
        }
        return null;
      };
    }
    if (newConfig.submitAfterSilence) {
      this._silenceSubmit = new SilenceSubmit(newConfig.submitAfterSilence, newConfig.stopAfterSubmit);
    }
    const serviceName = SpeechToText.getServiceName(newConfig);
    return {serviceName, processedConfig};
  }

  private static getServiceName(config: SpeechToTextConfig) {
    // --- WebSpeech API ---
    // https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API
    // --- Transformers JS ---
    // https://github.com/huggingface/transformers.js
    if (config.transformers) return 'transformers';
    return config.azure ? 'azure' : 'webspeech';
  }

  private buttonClick(textInput: TextInputEl, isInputEnabled: boolean, serviceName: string, config?: ProcessedConfig) {
    const events = config?.events;
    textInput.removePlaceholderStyle();

    if (serviceName === 'transformers' && config?.transformers) {
      this.handleTransformers(textInput, isInputEnabled, config.transformers, events);
      return;
    }

    SpeechToElement.toggle(serviceName as 'webspeech', {
      insertInCursorLocation: false,
      element: isInputEnabled ? textInput.inputElementRef : undefined,
      onError: () => {
        this.onError();
        this._silenceSubmit?.clearSilenceTimeout();
      },
      onStart: () => {
        this.changeToActive();
        events?.onStart?.();
      },
      onStop: () => {
        this._validationHandler?.();
        this._silenceSubmit?.clearSilenceTimeout();
        this.changeToDefault();
        events?.onStop?.();
      },
      onPauseTrigger: (isStart: boolean) => {
        this._silenceSubmit?.onPause(isStart, textInput, this.elementRef.onclick as () => void);
        events?.onPauseTrigger?.(isStart);
      },
      onPreResult: (text: string, isFinal: boolean) => {
        events?.onPreResult?.(text, isFinal);
      },
      onResult: (text: string, isFinal: boolean) => {
        if (isFinal) this._validationHandler?.();
        this._silenceSubmit?.resetSilenceTimeout(textInput, this.elementRef.onclick as () => void);
        events?.onResult?.(text, isFinal);
      },
      onCommandModeTrigger: (isStart: boolean) => {
        this.onCommandModeTrigger(isStart);
        events?.onCommandModeTrigger?.(isStart);
      },
      ...config,
    });
  }

  private handleTransformers(
    textInput: TextInputEl,
    isInputEnabled: boolean,
    transformersConfig: TransformersOptions,
    events?: SpeechEvents & { onError?: (error: unknown) => void }
  ) {
    // Only allow if input is enabled
    if (!isInputEnabled) return;

    // Start recording audio
    const chunks: BlobPart[] = [];
    let mediaRecorder: MediaRecorder | null = null;
    let worker: Worker | null = null;

    const startRecording = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
          const blob = new Blob(chunks, { type: 'audio/webm' });
          const arrayBuffer = await blob.arrayBuffer();

          // Start worker
          worker = new Worker(
            new URL('./transformersWorker.ts', import.meta.url),
            { type: 'module' }
          );
          worker.onmessage = (evt) => {
            const { text, error } = evt.data;
            if (error) {
              this.onError();
              events?.onError?.(error);
            } else if (text) {
              // Insert text into input
              (textInput.inputElementRef as HTMLInputElement).value = text;
              events?.onResult?.(text, true);
              this._validationHandler?.();
            }
            worker?.terminate();
          };

          worker.postMessage({
            audioBuffer: arrayBuffer,
            config: transformersConfig
          });
        };

        mediaRecorder.start();

        // Stop after 10 seconds or on button click again
        setTimeout(() => mediaRecorder?.state === 'recording' && mediaRecorder.stop(), 10000);

        // Optionally, allow stopping on button click again
        this.elementRef.onclick = () => {
          if (mediaRecorder?.state === 'recording') {
            mediaRecorder.stop();
          }
        };

        events?.onStart?.();
      } catch (err) {
        this.onError();
        events?.onError?.(err);
      }
    };

    startRecording();
  }

  private onCommandModeTrigger(isStart: boolean) {
    if (isStart) {
      this.changeToCommandMode();
    } else {
      this.changeToActive();
    }
  }

  private onError() {
    this._addErrorMessage('speechToText', 'speech input error');
  }

  public static toggleSpeechAfterSubmit(microphoneButton: HTMLElement, stopAfterSubmit = true) {
    microphoneButton.click();
    if (!stopAfterSubmit) setTimeout(() => microphoneButton.click(), SpeechToText.MICROPHONE_RESET_TIMEOUT_MS);
  }
}
