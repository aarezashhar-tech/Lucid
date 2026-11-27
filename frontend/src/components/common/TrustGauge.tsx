import React from "react";

export function TrustGauge({
  score,
  size = 120,
  strokeWidth = 10,
  showLabel = true,
}: {
  score: number | string;
  size?: number;
  strokeWidth?: number;
  showLabel?: boolean;
}) {
  const numericScore = typeof score === "number" ? Math.max(0, Math.min(100, score)) : parseInt(String(score) || "0", 10) || 0;

  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (numericScore / 100) * circumference;

  let strokeColor = "#22c55e"; // Emerald / Green >= 75
  let glowColor = "rgba(34, 197, 94, 0.25)";
  if (numericScore < 40) {
    strokeColor = "#ef4444"; // Ruby / Red < 40
    glowColor = "rgba(239, 68, 68, 0.25)";
  } else if (numericScore < 75) {
    strokeColor = "#f59e0b"; // Amber 40-74
    glowColor = "rgba(245, 158, 11, 0.25)";
  }

  return (
    <div className="relative inline-flex flex-col items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background track circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-surface-container opacity-50"
        />
        {/* Animated value stroke circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          style={{
            transition: "stroke-dashoffset 1s ease-out, stroke 0.4s ease",
            filter: `drop-shadow(0 0 6px ${glowColor})`,
          }}
        />
      </svg>
      {/* Center Score Display */}
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="font-extrabold tracking-tight text-on-surface tabular-nums" style={{ fontSize: size * 0.28, lineHeight: 1 }}>
          {numericScore}
        </span>
        {showLabel && (
          <span className="font-label-xs text-on-surface-variant uppercase tracking-wider mt-0.5" style={{ fontSize: Math.max(9, size * 0.09) }}>
            Trust Score
          </span>
        )}
      </div>
    </div>
  );
}
