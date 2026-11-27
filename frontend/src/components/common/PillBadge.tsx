import React from "react";
import { MatIcon } from "./MatIcon";
import { VerdictType } from "../../types/verification";

export function PillBadge({
  variant = "verdict",
  status,
  size = "md",
  className = "",
  showIcon = true,
}: {
  variant?: "verdict" | "stance" | "tier" | "generic";
  status: string;
  size?: "xs" | "sm" | "md" | "lg";
  className?: string;
  showIcon?: boolean;
}) {
  const norm = (status || "").toUpperCase();

  let bgClass = "bg-surface-container text-on-surface-variant border-surface-container-high";
  let iconName = "info";
  let label = status;

  if (variant === "verdict") {
    if (norm.includes("VERIFIED") || norm.includes("TRUE")) {
      bgClass = "bg-tertiary-container/20 text-tertiary border-tertiary/30 dark:bg-tertiary-container/30 dark:text-on-tertiary-fixed";
      iconName = "verified";
      label = norm.includes("LIKELY") ? "Likely True" : "Verified";
    } else if (norm.includes("REVIEW") || norm.includes("MIXED") || norm.includes("UNVERIFIED")) {
      bgClass = "bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30";
      iconName = "warning";
      label = norm.includes("REVIEW") ? "Needs Review" : norm.includes("MIXED") ? "Mixed Evidence" : "Unverified";
    } else if (norm.includes("FALSE") || norm.includes("FAKE") || norm.includes("SUSPICIOUS")) {
      bgClass = "bg-error-container/40 text-error border-error/30 dark:text-red-400";
      iconName = "cancel";
      label = norm.includes("LIKELY") ? "Likely False" : "False";
    }
  } else if (variant === "stance") {
    if (norm === "SUPPORT") {
      bgClass = "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30";
      iconName = "check_circle";
      label = "Supporting";
    } else if (norm === "CONTRADICT") {
      bgClass = "bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30";
      iconName = "cancel";
      label = "Contradicting";
    } else {
      bgClass = "bg-surface-container-high/60 text-on-surface-variant border-surface-container-high";
      iconName = "link";
      label = "Related / Neutral";
    }
  } else if (variant === "tier") {
    if (norm.includes("1") || norm.includes("WIRE") || norm.includes("HIGH")) {
      bgClass = "bg-primary/10 text-primary border-primary/25";
      iconName = "verified_user";
      label = "Tier 1 Wire";
    } else {
      bgClass = "bg-surface-container text-on-surface-variant border-surface-container-high";
      iconName = "feed";
      label = status;
    }
  }

  const sizeStyles = {
    xs: "px-2 py-0.5 text-[10px] font-semibold gap-1",
    sm: "px-2.5 py-1 text-xs font-medium gap-1.5",
    md: "px-3 py-1 text-xs sm:text-sm font-semibold gap-1.5",
    lg: "px-4 py-1.5 text-sm sm:text-base font-bold gap-2",
  };

  const iconSizes = {
    xs: 12,
    sm: 14,
    md: 16,
    lg: 18,
  };

  return (
    <span
      className={`inline-flex items-center rounded-full border shadow-2xs transition-colors ${sizeStyles[size]} ${bgClass} ${className}`}
    >
      {showIcon && <MatIcon name={iconName} size={iconSizes[size]} />}
      <span>{label}</span>
    </span>
  );
}
