import React, { useState, useRef, useEffect, useCallback } from "react";
import { MatIcon } from "../common/MatIcon";
import { LucidBrandLogo } from "../common/LucidBrandLogo";

export function DinoRunnerGame({ isDarkMode }: { isDarkMode: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [gameState, setGameState] = useState<"idle" | "playing" | "gameover">("idle");
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(() => {
    try {
      return parseInt(localStorage.getItem("lucid_dino_high_score") || "0", 10) || 0;
    } catch {
      return 0;
    }
  });

  const gameStateRef = useRef(gameState);
  gameStateRef.current = gameState;
  const scoreRef = useRef(score);
  scoreRef.current = score;
  const highScoreRef = useRef(highScore);
  highScoreRef.current = highScore;

  const playerRef = useRef({
    x: 45,
    y: 114,
    width: 22,
    height: 22,
    vy: 0,
    jumpForce: -10.5,
    gravity: 0.62,
    groundY: 114,
    isGrounded: true,
  });

  const obstaclesRef = useRef<Array<{ x: number; y: number; width: number; height: number }>>([]);
  const animFrameIdRef = useRef<number | null>(null);
  const groundOffsetRef = useRef(0);
  const frameCountRef = useRef(0);
  const speedRef = useRef(4.6);

  const jump = useCallback(() => {
    if (gameStateRef.current === "idle" || gameStateRef.current === "gameover") {
      obstaclesRef.current = [];
      scoreRef.current = 0;
      setScore(0);
      speedRef.current = 4.8;
      frameCountRef.current = 0;
      playerRef.current.y = playerRef.current.groundY;
      playerRef.current.vy = playerRef.current.jumpForce;
      playerRef.current.isGrounded = false;
      gameStateRef.current = "playing";
      setGameState("playing");
    } else if (gameStateRef.current === "playing") {
      if (playerRef.current.isGrounded) {
        playerRef.current.vy = playerRef.current.jumpForce;
        playerRef.current.isGrounded = false;
      }
    }
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" || e.code === "ArrowUp") {
        e.preventDefault();
        jump();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [jump]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let lastObstacleSpawn = 0;

    const loop = () => {
      const currentCanvas = canvasRef.current;
      if (!currentCanvas) return;
      const width = currentCanvas.width;
      const height = currentCanvas.height;
      const groundY = 136;

      ctx.clearRect(0, 0, width, height);

      const groundColor = isDarkMode ? "#334155" : "#c3c6d7";
      const groundDotColor = isDarkMode ? "#64748b" : "#94a3b8";
      const playerColor = isDarkMode ? "#38bdf8" : "#004ac6";
      const playerGlow = isDarkMode ? "rgba(56, 189, 248, 0.45)" : "rgba(0, 74, 198, 0.25)";
      const obstacleColor = isDarkMode ? "#f87171" : "#ba1a1a";

      if (gameStateRef.current === "playing") {
        frameCountRef.current += 1;
        if (frameCountRef.current % 5 === 0) {
          scoreRef.current += 1;
          setScore(scoreRef.current);
          if (scoreRef.current > highScoreRef.current) {
            highScoreRef.current = scoreRef.current;
            setHighScore(scoreRef.current);
            try {
              localStorage.setItem("lucid_dino_high_score", String(scoreRef.current));
            } catch {}
          }
          speedRef.current = 4.8 + Math.min(scoreRef.current / 250, 4.5);
        }

        const player = playerRef.current;
        player.vy += player.gravity;
        player.y += player.vy;
        if (player.y >= player.groundY) {
          player.y = player.groundY;
          player.vy = 0;
          player.isGrounded = true;
        }

        const timeSinceLastSpawn = frameCountRef.current - lastObstacleSpawn;
        const minGapFrames = Math.max(50, Math.floor(125 - speedRef.current * 7));
        if (timeSinceLastSpawn > minGapFrames && Math.random() < 0.035) {
          const obsHeight = 20 + Math.floor(Math.random() * 20);
          const obsWidth = 14 + Math.floor(Math.random() * 12);
          obstaclesRef.current.push({
            x: width + 20,
            y: groundY - obsHeight,
            width: obsWidth,
            height: obsHeight,
          });
          lastObstacleSpawn = frameCountRef.current;
        }

        const currentSpeed = speedRef.current;
        obstaclesRef.current.forEach((obs) => {
          obs.x -= currentSpeed;
        });

        obstaclesRef.current = obstaclesRef.current.filter((obs) => obs.x + obs.width > -20);
        groundOffsetRef.current = (groundOffsetRef.current + currentSpeed) % 24;

        const pLeft = player.x + 3;
        const pRight = player.x + player.width - 3;
        const pTop = player.y + 3;
        const pBottom = player.y + player.height - 1;

        for (const obs of obstaclesRef.current) {
          const oLeft = obs.x + 2;
          const oRight = obs.x + obs.width - 2;
          const oTop = obs.y + 2;
          const oBottom = obs.y + obs.height;

          if (pRight > oLeft && pLeft < oRight && pBottom > oTop && pTop < oBottom) {
            gameStateRef.current = "gameover";
            setGameState("gameover");
            break;
          }
        }
      }

      // Draw ground line
      ctx.beginPath();
      ctx.strokeStyle = groundColor;
      ctx.lineWidth = 2;
      ctx.moveTo(0, groundY);
      ctx.lineTo(width, groundY);
      ctx.stroke();

      // Ground dash dots
      ctx.fillStyle = groundDotColor;
      const dotSpacing = 24;
      for (let x = -groundOffsetRef.current; x < width; x += dotSpacing) {
        ctx.fillRect(x, groundY + 4, 6, 2);
        if ((x + 12) % 48 === 0) {
          ctx.fillRect(x + 10, groundY + 8, 3, 2);
        }
      }

      // Draw player box
      const p = playerRef.current;
      ctx.save();
      ctx.shadowColor = playerGlow;
      ctx.shadowBlur = 10;
      ctx.fillStyle = playerColor;
      ctx.beginPath();
      if ("roundRect" in ctx && typeof ctx.roundRect === "function") {
        ctx.roundRect(p.x, p.y, p.width, p.height, 4);
      } else {
        ctx.rect(p.x, p.y, p.width, p.height);
      }
      ctx.fill();

      // Eye dot
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(p.x + p.width - 7, p.y + 5, 3, 3);
      ctx.restore();

      // Draw obstacles
      obstaclesRef.current.forEach((obs) => {
        ctx.save();
        ctx.fillStyle = obstacleColor;
        ctx.beginPath();
        if ("roundRect" in ctx && typeof ctx.roundRect === "function") {
          ctx.roundRect(obs.x, obs.y, obs.width, obs.height, [3, 3, 0, 0]);
        } else {
          ctx.rect(obs.x, obs.y, obs.width, obs.height);
        }
        ctx.fill();
        ctx.fillStyle = isDarkMode ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.4)";
        ctx.fillRect(obs.x + 3, obs.y + 4, 3, obs.height - 6);
        ctx.restore();
      });

      // Text prompts
      if (gameStateRef.current === "idle") {
        ctx.fillStyle = isDarkMode ? "#f1f5f9" : "#131b2e";
        ctx.font = "bold 14px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Press Space or Tap to Jump", width / 2, groundY - 50);
        ctx.fillStyle = isDarkMode ? "#94a3b8" : "#5f6272";
        ctx.font = "12px Inter, sans-serif";
        ctx.fillText("Dodge the incoming obstacles!", width / 2, groundY - 28);
      } else if (gameStateRef.current === "gameover") {
        ctx.fillStyle = isDarkMode ? "#f87171" : "#ba1a1a";
        ctx.font = "bold 18px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("GAME OVER", width / 2, groundY - 60);

        ctx.fillStyle = isDarkMode ? "#f1f5f9" : "#131b2e";
        ctx.font = "bold 13px Inter, sans-serif";
        ctx.fillText(`Score: ${scoreRef.current}  |  Best: ${highScoreRef.current}`, width / 2, groundY - 36);

        ctx.fillStyle = isDarkMode ? "#94a3b8" : "#5f6272";
        ctx.font = "12px Inter, sans-serif";
        ctx.fillText("Press Space or Tap to Restart", width / 2, groundY - 14);
      }

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);
    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, [isDarkMode]);

  return (
    <div className="w-full flex flex-col items-center select-none">
      <div className="w-full flex items-center justify-between px-2 pb-2 text-xs font-mono font-semibold">
        <span className="text-on-surface-variant flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
          SCORE: <span className="text-primary font-bold">{score.toString().padStart(5, "0")}</span>
        </span>
        <span className="text-outline">
          HIGH: <span className="font-bold text-on-surface">{highScore.toString().padStart(5, "0")}</span>
        </span>
      </div>

      <div
        onClick={jump}
        onTouchStart={(e) => {
          e.preventDefault();
          jump();
        }}
        className="w-full rounded-xl bg-surface-container-low border border-surface-container-high/60 cursor-pointer overflow-hidden touch-none relative"
        style={{ height: "160px" }}
        role="button"
        tabIndex={0}
        aria-label="Offline Mini Game: Tap or press space to jump"
      >
        <canvas
          ref={canvasRef}
          width={580}
          height={160}
          className="w-full h-full block"
        />
      </div>
      <p className="mt-2 text-xs text-outline text-center">
        Controls: <kbd className="px-1.5 py-0.5 rounded bg-surface-container text-on-surface font-mono">Space</kbd> or <kbd className="px-1.5 py-0.5 rounded bg-surface-container text-on-surface font-mono">↑</kbd> to jump · Tap on mobile
      </p>
    </div>
  );
}

export function Offline404Overlay({
  onRetry,
  onDismiss,
  isChecking,
  isDarkMode,
}: {
  onRetry: () => void;
  onDismiss: () => void;
  isChecking: boolean;
  isDarkMode: boolean;
}) {
  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto flex flex-col items-center justify-center p-4 sm:p-6 bg-surface/95 backdrop-blur-md transition-all animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-label="Offline 404 Notice"
    >
      <div className="max-w-xl w-full mx-auto flex flex-col items-center text-center">
        <div className="mb-2">
          <LucidBrandLogo size="md" showText={false} />
        </div>

        <h1 className="font-display-hero text-6xl sm:text-7xl md:text-8xl font-black text-primary tracking-tight leading-none mb-1">
          404
        </h1>

        <h2 className="text-2xl sm:text-3xl font-bold text-on-surface tracking-tight mt-1">
          Oops, You Are Offline!
        </h2>

        <p className="font-body-md text-xs sm:text-sm text-on-surface-variant max-w-md mt-2 mb-5 leading-relaxed">
          The internet connection was lost. Play the mini-game below while waiting for connection.
        </p>

        <div className="w-full lucid-card bg-surface-container-lowest rounded-2xl shadow-xl border border-surface-container-high/50 p-4 sm:p-5 mb-5 transition-all">
          <DinoRunnerGame isDarkMode={isDarkMode} />
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 w-full sm:w-auto">
          <button
            onClick={onRetry}
            disabled={isChecking}
            className="w-full sm:w-auto min-h-[44px] px-6 py-2.5 rounded-full bg-primary text-on-primary font-semibold text-sm hover:opacity-95 transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm disabled:opacity-60"
            type="button"
            aria-label="Retry network connection"
          >
            <MatIcon name="refresh" size={18} className={isChecking ? "animate-spin" : ""} />
            <span>{isChecking ? "Checking Connection..." : "Retry Connection"}</span>
          </button>

          <button
            onClick={onDismiss}
            className="w-full sm:w-auto min-h-[44px] px-6 py-2.5 rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface font-semibold text-sm transition-all flex items-center justify-center gap-2 cursor-pointer border border-surface-container-high/40"
            type="button"
            aria-label="Dismiss offline screen and view cached page"
          >
            <MatIcon name="visibility" size={18} />
            <span>Dismiss / View Cached Page</span>
          </button>
        </div>
      </div>
    </div>
  );
}
