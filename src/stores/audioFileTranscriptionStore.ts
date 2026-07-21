import { open } from "@tauri-apps/plugin-dialog";
import { create } from "zustand";
import { commands, type HistoryEntry } from "@/bindings";

const AUDIO_FILE_EXTENSIONS = ["wav", "mp3", "m4a", "mp4", "flac", "ogg"];

interface ChooseAudioFileLabels {
  dialogTitle: string;
  audioFiles: string;
}

type AudioFileTranscriptionStatus =
  | "idle"
  | "transcribing"
  | "cancelling"
  | "completed"
  | "error";

type AudioFileTranscriptionErrorKind =
  | "generic"
  | "liveTranscriptionInProgress";

const LIVE_TRANSCRIPTION_IN_PROGRESS_ERROR =
  "Another transcription is already in progress";

const messageFromError = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);

const errorKindFromMessage = (
  message: string,
): AudioFileTranscriptionErrorKind =>
  message.includes(LIVE_TRANSCRIPTION_IN_PROGRESS_ERROR)
    ? "liveTranscriptionInProgress"
    : "generic";

interface AudioFileTranscriptionStore {
  selectedPath: string | null;
  status: AudioFileTranscriptionStatus;
  completedEntry: HistoryEntry | null;
  errorMessage: string | null;
  errorKind: AudioFileTranscriptionErrorKind | null;
  chooseAudioFile: (labels: ChooseAudioFileLabels) => Promise<void>;
  transcribeAudioFile: (path: string) => Promise<void>;
  cancelTranscription: () => Promise<void>;
  clearError: () => void;
  clearResult: () => void;
}

export const useAudioFileTranscriptionStore =
  create<AudioFileTranscriptionStore>((set, get) => ({
    selectedPath: null,
    status: "idle",
    completedEntry: null,
    errorMessage: null,
    errorKind: null,

    chooseAudioFile: async ({ dialogTitle, audioFiles }) => {
      const { status } = get();
      if (status === "transcribing" || status === "cancelling") return;

      let path: string | string[] | null;
      try {
        path = await open({
          multiple: false,
          directory: false,
          title: dialogTitle,
          filters: [
            {
              name: audioFiles,
              extensions: AUDIO_FILE_EXTENSIONS,
            },
          ],
        });
      } catch (dialogError) {
        console.error("Failed to open the audio file picker:", dialogError);
        const message = messageFromError(dialogError);
        set({
          status: "error",
          errorMessage: message,
          errorKind: "generic",
        });
        return;
      }

      if (typeof path !== "string") return;
      await get().transcribeAudioFile(path);
    },

    transcribeAudioFile: async (path) => {
      const { status } = get();
      if (status === "transcribing" || status === "cancelling") return;

      set({
        selectedPath: path,
        status: "transcribing",
        completedEntry: null,
        errorMessage: null,
        errorKind: null,
      });

      try {
        const result = await commands.transcribeAudioFile(path);
        if (result.status === "error") {
          throw new Error(String(result.error));
        }
        set({
          status: "completed",
          completedEntry: result.data,
        });
      } catch (transcriptionError) {
        console.error(
          "Failed to transcribe imported audio file:",
          transcriptionError,
        );
        if (get().status === "cancelling") {
          set({ status: "idle", errorMessage: null, errorKind: null });
        } else {
          const message = messageFromError(transcriptionError);
          set({
            status: "error",
            errorMessage: message,
            errorKind: errorKindFromMessage(message),
          });
        }
      }
    },

    cancelTranscription: async () => {
      if (get().status !== "transcribing") return;

      set({ status: "cancelling", errorMessage: null, errorKind: null });
      try {
        await commands.cancelOperation();
      } catch (cancelError) {
        console.error(
          "Failed to cancel imported audio transcription:",
          cancelError,
        );
        if (get().status === "cancelling") {
          const message = messageFromError(cancelError);
          set({
            status: "transcribing",
            errorMessage: message,
            errorKind: "generic",
          });
        }
      }
    },

    clearError: () => {
      set({ errorMessage: null, errorKind: null });
    },

    clearResult: () => {
      set({
        selectedPath: null,
        status: "idle",
        completedEntry: null,
        errorMessage: null,
        errorKind: null,
      });
    },
  }));
