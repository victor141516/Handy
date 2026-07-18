import {
  CheckCircle2,
  FileAudio2,
  History,
  Loader2,
  Upload,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import type { HistoryEntry } from "@/bindings";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";
import { useAudioFileTranscriptionStore } from "@/stores/audioFileTranscriptionStore";

interface AudioFileTranscriptionProps {
  onOpenHistory?: () => void;
}

const fileNameFromPath = (path: string): string =>
  path.split(/[/\\]/).pop() || path;

interface AudioFilePickerCardProps {
  selectedPath: string | null;
  isTranscribing: boolean;
  isCancelling: boolean;
  onChoose: () => void;
  onCancel: () => void;
}

const AudioFilePickerCard: React.FC<AudioFilePickerCardProps> = ({
  selectedPath,
  isTranscribing,
  isCancelling,
  onChoose,
  onCancel,
}) => {
  const { t } = useTranslation();

  return (
    <div className="bg-background border border-mid-gray/20 rounded-lg overflow-hidden">
      <div className="flex flex-col items-center gap-4 px-6 py-8 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-logo-primary/15 text-logo-primary">
          <FileAudio2 className="h-7 w-7" />
        </div>

        <div className="space-y-1">
          <p className="text-sm font-medium">
            {selectedPath
              ? fileNameFromPath(selectedPath)
              : t("audioFileTranscription.noFileSelected")}
          </p>
          <p className="text-xs text-mid-gray">
            {t("audioFileTranscription.supportedFormats")}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            type="button"
            onClick={onChoose}
            disabled={isTranscribing}
            className="flex items-center gap-2"
          >
            {isTranscribing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Upload className="h-4 w-4" />
            )}
            {isTranscribing
              ? t("audioFileTranscription.transcribing")
              : selectedPath
                ? t("audioFileTranscription.chooseAnother")
                : t("audioFileTranscription.chooseFile")}
          </Button>

          {isTranscribing && (
            <Button
              type="button"
              variant="secondary"
              onClick={onCancel}
              disabled={isCancelling}
              className="flex items-center gap-2"
            >
              {isCancelling ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <X className="h-4 w-4" />
              )}
              {isCancelling
                ? t("audioFileTranscription.cancelling")
                : t("audioFileTranscription.cancel")}
            </Button>
          )}
        </div>

        {isTranscribing && (
          <p className="text-xs text-mid-gray">
            {t("audioFileTranscription.keepOpen")}
          </p>
        )}
      </div>
    </div>
  );
};

const TranscriptionResult: React.FC<{
  entry: HistoryEntry;
  onOpenHistory?: () => void;
}> = ({ entry, onOpenHistory }) => {
  const { t } = useTranslation();

  return (
    <div className="space-y-2">
      <Alert variant="success">{t("audioFileTranscription.completed")}</Alert>

      <div className="bg-background border border-mid-gray/20 rounded-lg px-4 py-4 space-y-4">
        <div className="flex items-center gap-2 text-sm font-medium">
          <CheckCircle2 className="h-4 w-4 text-green-500" />
          {t("audioFileTranscription.result")}
        </div>
        <p className="select-text cursor-text whitespace-pre-wrap break-words text-sm text-text/90">
          {entry.transcription_text}
        </p>
        <div className="flex justify-end">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            className="flex items-center gap-2"
            onClick={onOpenHistory}
          >
            <History className="h-4 w-4" />
            {t("audioFileTranscription.openHistory")}
          </Button>
        </div>
      </div>
    </div>
  );
};

export const AudioFileTranscription: React.FC<AudioFileTranscriptionProps> = ({
  onOpenHistory,
}) => {
  const { t } = useTranslation();
  const selectedPath = useAudioFileTranscriptionStore(
    (state) => state.selectedPath,
  );
  const status = useAudioFileTranscriptionStore((state) => state.status);
  const errorMessage = useAudioFileTranscriptionStore(
    (state) => state.errorMessage,
  );
  const completedEntry = useAudioFileTranscriptionStore(
    (state) => state.completedEntry,
  );
  const chooseAudioFile = useAudioFileTranscriptionStore(
    (state) => state.chooseAudioFile,
  );
  const cancelTranscription = useAudioFileTranscriptionStore(
    (state) => state.cancelTranscription,
  );
  const isTranscribing = status === "transcribing" || status === "cancelling";
  const isCancelling = status === "cancelling";

  const handleChoose = () =>
    chooseAudioFile({
      dialogTitle: t("audioFileTranscription.dialogTitle"),
      audioFiles: t("audioFileTranscription.audioFiles"),
    });

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <div className="space-y-2">
        <div className="px-4">
          <h2 className="text-xs font-medium text-mid-gray uppercase tracking-wide">
            {t("audioFileTranscription.title")}
          </h2>
          <p className="text-xs text-mid-gray mt-1">
            {t("audioFileTranscription.description")}
          </p>
        </div>

        <AudioFilePickerCard
          selectedPath={selectedPath}
          isTranscribing={isTranscribing}
          isCancelling={isCancelling}
          onChoose={handleChoose}
          onCancel={cancelTranscription}
        />
      </div>

      {errorMessage && (
        <Alert variant="error">{t("audioFileTranscription.error")}</Alert>
      )}

      {completedEntry && (
        <TranscriptionResult
          entry={completedEntry}
          onOpenHistory={onOpenHistory}
        />
      )}
    </div>
  );
};
