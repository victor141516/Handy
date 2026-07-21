import { useEffect, useRef, useState } from "react";
import {
  Check,
  CheckCircle2,
  Copy,
  FileAudio2,
  History,
  Loader2,
  RotateCcw,
  Sparkles,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { commands, type HistoryEntry, type LLMPrompt } from "@/bindings";
import { Button } from "@/components/ui/Button";
import { useSettings } from "@/hooks/useSettings";
import { useAudioFileTranscriptionStore } from "@/stores/audioFileTranscriptionStore";

interface AudioFileTranscriptionProps {
  onOpenHistory?: () => void;
}

const fileNameFromPath = (path: string): string =>
  path.split(/[/\\]/).pop() || path;

const IconButton: React.FC<{
  onClick: () => void;
  title: string;
  disabled?: boolean;
  active?: boolean;
  children: React.ReactNode;
}> = ({ onClick, title, disabled, active, children }) => (
  <button
    type="button"
    onClick={onClick}
    disabled={disabled}
    className={`p-1.5 rounded-md flex items-center justify-center transition-colors cursor-pointer disabled:cursor-not-allowed disabled:text-text/20 ${
      active
        ? "text-logo-primary hover:text-logo-primary/80"
        : "text-text/50 hover:text-logo-primary"
    }`}
    title={title}
  >
    {children}
  </button>
);

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
  selectedPath: string | null;
  postProcessEnabled: boolean;
  postProcessPrompts: LLMPrompt[];
  onRetranscribe: () => void;
  onClearResult: () => void;
  onOpenHistory?: () => void;
}> = ({
  entry,
  selectedPath,
  postProcessEnabled,
  postProcessPrompts,
  onRetranscribe,
  onClearResult,
  onOpenHistory,
}) => {
  const { t } = useTranslation();
  const [showCopied, setShowCopied] = useState(false);
  const [postProcessingPromptId, setPostProcessingPromptId] = useState<
    string | null
  >(null);
  const [postProcessMenuOpen, setPostProcessMenuOpen] = useState(false);
  const postProcessMenuRef = useRef<HTMLDivElement>(null);
  const isPostProcessing = postProcessingPromptId !== null;
  const hasPostProcessPrompts = postProcessPrompts.length > 0;

  useEffect(() => {
    if (!postProcessEnabled) {
      setPostProcessMenuOpen(false);
    }
  }, [postProcessEnabled]);

  useEffect(() => {
    if (!postProcessMenuOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (
        postProcessMenuRef.current &&
        !postProcessMenuRef.current.contains(event.target as Node)
      ) {
        setPostProcessMenuOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [postProcessMenuOpen]);

  const handleCopyText = async () => {
    try {
      await navigator.clipboard.writeText(entry.transcription_text);
      setShowCopied(true);
      setTimeout(() => setShowCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy imported audio transcription:", error);
    }
  };

  const handlePostProcess = async (prompt: LLMPrompt) => {
    try {
      setPostProcessMenuOpen(false);
      setPostProcessingPromptId(prompt.id);
      const result = await commands.postProcessHistoryEntry(
        entry.id,
        prompt.id,
      );
      if (result.status !== "ok") {
        throw new Error(String(result.error));
      }
      toast.success(t("settings.history.postProcessCopied"));
    } catch (error) {
      console.error(
        "Failed to post-process imported audio transcription:",
        error,
      );
      toast.error(t("settings.history.postProcessError"));
    } finally {
      setPostProcessingPromptId(null);
    }
  };

  return (
    <div className="space-y-2">
      <div className="bg-background border border-mid-gray/20 rounded-lg px-4 py-4 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            {t("audioFileTranscription.result")}
          </div>

          <div className="flex shrink-0 items-center">
            <IconButton
              onClick={handleCopyText}
              disabled={isPostProcessing}
              title={t("settings.history.copyToClipboard")}
            >
              {showCopied ? (
                <Check width={16} height={16} />
              ) : (
                <Copy width={16} height={16} />
              )}
            </IconButton>
            <IconButton
              onClick={onRetranscribe}
              disabled={!selectedPath || isPostProcessing}
              title={t("settings.history.retranscribe")}
            >
              <RotateCcw width={16} height={16} />
            </IconButton>
            {postProcessEnabled && (
              <div className="relative" ref={postProcessMenuRef}>
                <IconButton
                  onClick={() => setPostProcessMenuOpen((open) => !open)}
                  disabled={isPostProcessing || !hasPostProcessPrompts}
                  active={postProcessMenuOpen || isPostProcessing}
                  title={
                    hasPostProcessPrompts
                      ? t("settings.history.postProcess")
                      : t("settings.history.postProcessNoPrompts")
                  }
                >
                  <span
                    className="relative flex h-4 w-4 items-center justify-center"
                    aria-hidden="true"
                  >
                    {isPostProcessing && (
                      <style>{`
                        @keyframes audio-file-post-process-pulse {
                          0%, 100% { opacity: 0.65; transform: scale(0.92); filter: drop-shadow(0 0 0 currentColor); }
                          50% { opacity: 1; transform: scale(1.08); filter: drop-shadow(0 0 5px currentColor); }
                        }
                      `}</style>
                    )}
                    <Sparkles
                      width={16}
                      height={16}
                      className="shrink-0"
                      style={
                        isPostProcessing
                          ? {
                              animation:
                                "audio-file-post-process-pulse 1.1s ease-in-out infinite",
                            }
                          : undefined
                      }
                    />
                  </span>
                </IconButton>
                {postProcessMenuOpen && (
                  <div className="absolute right-0 top-full z-20 mt-1 min-w-44 overflow-hidden rounded-md border border-mid-gray/20 bg-background shadow-lg">
                    {postProcessPrompts.map((prompt) => (
                      <button
                        key={prompt.id}
                        type="button"
                        className="block w-full px-3 py-2 text-left text-xs text-text hover:bg-mid-gray/10 focus:bg-mid-gray/20 focus:outline-none"
                        onClick={() => handlePostProcess(prompt)}
                        disabled={isPostProcessing}
                      >
                        <span className="block truncate">{prompt.name}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            <IconButton
              onClick={onClearResult}
              disabled={isPostProcessing}
              title={t("audioFileTranscription.clearResult")}
            >
              <Trash2 width={16} height={16} />
            </IconButton>
          </div>
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
  const { getSetting } = useSettings();
  const selectedPath = useAudioFileTranscriptionStore(
    (state) => state.selectedPath,
  );
  const status = useAudioFileTranscriptionStore((state) => state.status);
  const errorMessage = useAudioFileTranscriptionStore(
    (state) => state.errorMessage,
  );
  const errorKind = useAudioFileTranscriptionStore((state) => state.errorKind);
  const completedEntry = useAudioFileTranscriptionStore(
    (state) => state.completedEntry,
  );
  const chooseAudioFile = useAudioFileTranscriptionStore(
    (state) => state.chooseAudioFile,
  );
  const cancelTranscription = useAudioFileTranscriptionStore(
    (state) => state.cancelTranscription,
  );
  const transcribeAudioFile = useAudioFileTranscriptionStore(
    (state) => state.transcribeAudioFile,
  );
  const clearError = useAudioFileTranscriptionStore(
    (state) => state.clearError,
  );
  const clearResult = useAudioFileTranscriptionStore(
    (state) => state.clearResult,
  );
  const isTranscribing = status === "transcribing" || status === "cancelling";
  const isCancelling = status === "cancelling";
  const previousStatusRef = useRef(status);
  const postProcessPrompts = getSetting("post_process_prompts") || [];
  const postProcessEnabled = getSetting("post_process_enabled") || false;

  const handleChoose = () =>
    chooseAudioFile({
      dialogTitle: t("audioFileTranscription.dialogTitle"),
      audioFiles: t("audioFileTranscription.audioFiles"),
    });

  const handleRetranscribe = () => {
    if (selectedPath) {
      transcribeAudioFile(selectedPath);
    }
  };

  useEffect(() => {
    const previousStatus = previousStatusRef.current;
    if (
      status === "completed" &&
      previousStatus !== "completed" &&
      completedEntry
    ) {
      toast.success(t("audioFileTranscription.completed"), { duration: 3000 });
    }
    previousStatusRef.current = status;
  }, [completedEntry, status, t]);

  useEffect(() => {
    if (!errorMessage) return;

    toast.error(
      t(
        errorKind === "liveTranscriptionInProgress"
          ? "audioFileTranscription.liveTranscriptionInProgress"
          : "audioFileTranscription.error",
      ),
      { duration: 5000 },
    );
    clearError();
  }, [clearError, errorKind, errorMessage, t]);

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

      {completedEntry && (
        <TranscriptionResult
          entry={completedEntry}
          selectedPath={selectedPath}
          postProcessEnabled={postProcessEnabled}
          postProcessPrompts={postProcessPrompts}
          onRetranscribe={handleRetranscribe}
          onClearResult={clearResult}
          onOpenHistory={onOpenHistory}
        />
      )}
    </div>
  );
};
