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

interface AudioFileTranscriptionStore {
  selectedPath: string | null;
  status: AudioFileTranscriptionStatus;
  completedEntry: HistoryEntry | null;
  errorMessage: string | null;
  chooseAudioFile: (labels: ChooseAudioFileLabels) => Promise<void>;
  transcribeAudioFile: (path: string) => Promise<void>;
  cancelTranscription: () => Promise<void>;
}

export const useAudioFileTranscriptionStore =
  create<AudioFileTranscriptionStore>((set, get) => ({
    selectedPath: null,
    status: "idle",
    completedEntry: null,
    errorMessage: null,

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
        set({
          status: "error",
          errorMessage: String(dialogError),
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
          set({ status: "idle", errorMessage: null });
        } else {
          set({
            status: "error",
            errorMessage: String(transcriptionError),
          });
        }
      }
    },

    cancelTranscription: async () => {
      if (get().status !== "transcribing") return;

      set({ status: "cancelling", errorMessage: null });
      try {
        await commands.cancelOperation();
      } catch (cancelError) {
        console.error(
          "Failed to cancel imported audio transcription:",
          cancelError,
        );
        if (get().status === "cancelling") {
          set({
            status: "transcribing",
            errorMessage: String(cancelError),
          });
        }
      }
    },
  }));
