import React, { useRef, useState } from "react";
import { MatIcon } from "../common/MatIcon";

export function DropZone({
  mode,
  selectedFile,
  onFileSelect,
  className = "",
}: {
  mode: "image" | "pdf";
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
  className?: string;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [dropError, setDropError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const acceptedTypes = mode === "image" ? ".png,.jpg,.jpeg,.webp,.avif" : ".pdf";
  const labelText =
    mode === "image"
      ? "Upload News Screenshot or Image (PNG, JPG)"
      : "Upload News Dossier or Document (PDF)";

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const validateAndSetFile = (file: File) => {
    setDropError(null);
    if (file.size > 10 * 1024 * 1024) {
      setDropError("File size exceeds 10MB limit. Please upload a smaller file.");
      return;
    }

    if (mode === "image" && !file.type.startsWith("image/")) {
      setDropError("Please upload a valid image file (PNG, JPG, WebP).");
      return;
    }

    if (mode === "pdf" && !file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") {
      setDropError("Please upload a valid PDF document.");
      return;
    }

    onFileSelect(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  return (
    <div className={`w-full ${className}`}>
      <input
        ref={inputRef}
        type="file"
        accept={acceptedTypes}
        onChange={handleInputChange}
        className="hidden"
      />

      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`group cursor-pointer rounded-2xl border-2 border-dashed p-6 sm:p-8 text-center transition-all duration-200 flex flex-col items-center justify-center gap-3 ${
          isDragging
            ? "border-primary bg-primary/5 scale-[1.01]"
            : selectedFile
            ? "border-primary/40 bg-surface-container-low"
            : "border-surface-container-high hover:border-primary/50 hover:bg-surface-container-low"
        }`}
      >
        <div
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-transform group-hover:scale-105 shadow-xs ${
            selectedFile ? "bg-primary text-on-primary" : "bg-surface-container text-primary"
          }`}
        >
          <MatIcon name={mode === "image" ? "image" : "picture_as_pdf"} size={24} />
        </div>

        {selectedFile ? (
          <div className="space-y-1">
            <p className="font-headline-sm text-sm sm:text-base font-semibold text-on-surface truncate max-w-xs sm:max-w-md">
              {selectedFile.name}
            </p>
            <p className="font-body-sm text-xs text-on-surface-variant">
              {(selectedFile.size / 1024).toFixed(1)} KB • Click or drop to replace
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            <p className="font-headline-sm text-sm sm:text-base font-semibold text-on-surface">
              {labelText}
            </p>
            <p className="font-body-sm text-xs text-on-surface-variant">
              Drag &amp; drop file here, or click to browse (Max 10MB)
            </p>
          </div>
        )}
      </div>

      {dropError && (
        <div className="mt-2.5 p-3 rounded-xl bg-error-container text-error text-xs flex items-center gap-2">
          <MatIcon name="error_outline" size={16} />
          <span>{dropError}</span>
        </div>
      )}
    </div>
  );
}
