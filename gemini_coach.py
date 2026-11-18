#!/usr/bin/env python3
"""Gemini Coach UCI Engine Wrapper.

A Universal Chess Interface (UCI) wrapper that combines local Stockfish
tactical evaluation with the Google Gemini API. Stockfish generates a short
list of strong candidate moves, and Gemini picks the one that best matches the
user's custom instructions while explaining it in natural language.

Design notes:
    * Gemini can only choose among Stockfish's candidates, so it can never
      play a move that is much worse than the best one.
    * Every failure path (no key, timeout, HTTP error, bad output, low clock,
      ``stop``) falls back to Stockfish's best move and says why in an
      ``info string``.
    * ``go`` runs in a worker thread, so ``isready``, ``stop`` and ``quit``
      are answered while the engine is "thinking".

Environment variables:
    GEMINI_API_KEY          Required for LLM moves; otherwise Stockfish plays.
    GEMINI_MODEL            Override the model ID (default below).
    GEMINI_THINKING_BUDGET  Optional integer sent as thinkingConfig.thinkingBudget
                            (e.g. 0 to disable thinking on models that allow it).
    STOCKFISH_PATH          Path to the Stockfish binary.
    COMMENT_LANGUAGE        Language of the commentary (default: Italian).
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import requests
import urllib3.util.connection as urllib3_cn

# Force IPv4 for outbound HTTP requests to prevent connection timeouts
# on networks with broken or slow IPv6 routing (common on some Linux setups).
# Note: this is process-global, which is fine for a single-purpose script.
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
STOCKFISH_PATH: str = (
    os.environ.get("STOCKFISH_PATH")
    or shutil.which("stockfish")
    or "/usr/bin/stockfish"
)

# Read on every turn so instructions can be updated mid-game.
INSTRUCTIONS_FILE: Path = Path.home() / ".config" / "gemini_chess_prompt.txt"

API_KEY: str = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
COMMENT_LANGUAGE: str = os.environ.get("COMMENT_LANGUAGE", "Italian")


def _env_int(name: str) -> int | None:
    """Read an optional integer environment variable (None if unset/invalid)."""
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


GEMINI_THINKING_BUDGET: int | None = _env_int("GEMINI_THINKING_BUDGET")

# Network / timing
API_TIMEOUT_SECONDS: float = 12.0          # hard cap for the whole API call
API_CONNECT_TIMEOUT_SECONDS: float = 4.0
MIN_API_BUDGET_SECONDS: float = 1.5        # below this, skip the API entirely
MOVE_OVERHEAD_SECONDS: float = 0.3         # safety margin for GUI/pipe latency
DEFAULT_MOVES_TO_GO: int = 30

# Candidate selection
STOCKFISH_THINK_TIME: float = 0.15
CANDIDATE_MOVES_COUNT: int = 6
MAX_CANDIDATE_LOSS_CP: int = 90            # drop moves worse than best by more
MATE_SCORE: int = 100_000

# Prompting
HISTORY_PLIES: int = 8
MAX_OUTPUT_TOKENS: int = 1024              # headroom in case thinking tokens count

UCI_MOVE_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

FALLBACK_NOTES: dict[str, str] = {
    "Italian": "Mossa di riserva calcolata da Stockfish ({reason}).",
    "English": "Fallback move calculated by Stockfish ({reason}).",
}


def log(message: str) -> None:
    """Write a diagnostic line to stderr (never stdout, which is the UCI channel)."""
    sys.stderr.write(message.rstrip("\n") + "\n")
    sys.stderr.flush()


def fallback_note(reason: str) -> str:
    """Build the commentary line shown when Stockfish's move is used instead."""
    template = FALLBACK_NOTES.get(COMMENT_LANGUAGE, FALLBACK_NOTES["English"])
    return template.format(reason=reason)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Raised when the Gemini step fails; the message is a short reason."""


@dataclass(frozen=True)
class Candidate:
    """A Stockfish candidate move."""

    uci: str
    san: str
    cp: int        # centipawns from the mover's point of view (mates are huge)
    score: str     # human-readable score, e.g. "+35" or "#3"
    line: str      # first plies of the principal variation, in SAN

    def describe(self) -> str:
        return f"{self.uci} ({self.san}) | eval: {self.score} | line: {self.line}"


@dataclass
class GoParams:
    """Parsed arguments of a UCI ``go`` command."""

    wtime: int | None = None
    btime: int | None = None
    winc: int | None = None
    binc: int | None = None
    movestogo: int | None = None
    movetime: int | None = None
    infinite: bool = False


# ---------------------------------------------------------------------------
# Pure helpers (easy to unit test)
# ---------------------------------------------------------------------------
_GO_INT_PARAMS = {"wtime", "btime", "winc", "binc", "movestogo", "movetime"}


def parse_go(tokens: list[str]) -> GoParams:
    """Parse the tokens of a ``go`` command, ignoring anything unsupported."""
    params = GoParams()
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "infinite":
            params.infinite = True
        elif token in _GO_INT_PARAMS and i + 1 < len(tokens):
            try:
                setattr(params, token, int(tokens[i + 1]))
            except ValueError:
                pass
            i += 1
        i += 1
    return params


def compute_move_budget(params: GoParams, turn: chess.Color) -> float | None:
    """Return the seconds we may spend on this move, or None if unconstrained."""
    if params.movetime is not None:
        return max(0.0, params.movetime / 1000 - MOVE_OVERHEAD_SECONDS)

    white = turn == chess.WHITE
    remaining_ms = params.wtime if white else params.btime
    if remaining_ms is None:
        return None

    increment_ms = (params.winc if white else params.binc) or 0
    remaining = remaining_ms / 1000
    moves_to_go = max(params.movestogo or DEFAULT_MOVES_TO_GO, 1)

    budget = remaining / moves_to_go + 0.8 * increment_ms / 1000
    budget = min(budget, remaining * 0.5)  # never plan to burn half the clock
    return max(0.0, budget - MOVE_OVERHEAD_SECONDS)


def parse_position(tokens: list[str]) -> chess.Board:
    """Build a board from a UCI ``position`` command.

    Supports ``position startpos [moves ...]`` and
    ``position fen <fen> [moves ...]``.

    Raises:
        ValueError: on a malformed command, FEN, or illegal move. The caller's
            current board is never touched, so bad input cannot corrupt state.
    """
    if len(tokens) < 2:
        raise ValueError("position: missing arguments")

    if tokens[1] == "startpos":
        board = chess.Board()
        rest = 2
    elif tokens[1] == "fen":
        end = tokens.index("moves") if "moves" in tokens else len(tokens)
        board = chess.Board(" ".join(tokens[2:end]))
        rest = end
    else:
        raise ValueError(f"position: unknown argument '{tokens[1]}'")

    if rest < len(tokens) and tokens[rest] == "moves":
        for uci_move in tokens[rest + 1:]:
            board.push_uci(uci_move)
    return board


def extract_uci_move(line: str) -> str | None:
    """Find the first UCI-shaped token in a line (tolerates 'Move: e2e4.')."""
    match = UCI_MOVE_RE.search(line.lower())
    return match.group(0) if match else None


def format_recent_moves(board: chess.Board, plies: int = HISTORY_PLIES) -> str:
    """Return the last few moves in numbered SAN, e.g. '4. Nf3 4... Nc6'."""
    replay = board.root()
    labelled: list[str] = []
    for move in board.move_stack:
        number = replay.fullmove_number
        label = f"{number}." if replay.turn == chess.WHITE else f"{number}..."
        labelled.append(f"{label} {replay.san(move)}")
        replay.push(move)
    return " ".join(labelled[-plies:])


def get_custom_instructions() -> str:
    """Read custom playstyle instructions from disk, with a sensible default."""
    try:
        if INSTRUCTIONS_FILE.is_file():
            content = INSTRUCTIONS_FILE.read_text(encoding="utf-8").strip()
            if content:
                return content
    except (OSError, UnicodeDecodeError) as exc:
        log(f"Could not read instructions file: {exc}")

    return (
        "Play solidly and pedagogically, prioritizing clear development "
        "and explaining the strategic plan behind each move."
    )


def build_prompt(
    board: chess.Board,
    candidates: list[Candidate],
    custom_prompt: str,
) -> str:
    """Assemble the Gemini prompt."""
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    history = format_recent_moves(board) or "(none)"
    candidate_block = "\n".join(c.describe() for c in candidates)

    return f"""
You are an expert chess coach and a UCI chess engine. Select ONE move to play in the current position and explain it clearly.

User instructions for this game:
"{custom_prompt}"

Current FEN: {board.fen()}
Recent moves: {history}
Side to move: {side_to_move}
Candidate moves (with centipawn evaluation and principal variation). You MUST choose one of these:
{candidate_block}

Strictly follow this output format:
Line 1: ONLY the chosen move in pure UCI format, copied from the candidates (e.g., e2e4 or g1f3). No backticks, no punctuation, no extra text.
From Line 2 onward: write an insightful analysis in {COMMENT_LANGUAGE} (4-5 prose sentences, no numbered lists or Markdown symbols) covering:
- The main strategic or tactical idea behind the chosen move.
- What threats it creates or what opponent weakness it exploits.
- How the move affects piece coordination, key square control, and overall board structure.
- How it fits with the recent moves, if relevant.
""".strip()


def describe_request_error(exc: Exception) -> str:
    """Turn a request exception into a short reason (never includes URLs)."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.ConnectionError):
        return "connection error"
    if isinstance(exc, ValueError):
        return "invalid JSON response"
    return type(exc).__name__


def extract_response_text(data: dict) -> str:
    """Join all text parts of the first Gemini candidate.

    Raises:
        LLMError: if the response has no usable text.
    """
    try:
        candidate = data["candidates"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("no candidates in response (blocked?)") from exc

    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and not part.get("thought")
    )
    if not text.strip():
        raise LLMError(f"empty response (finishReason={candidate.get('finishReason')})")
    return text


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class CoachEngine:
    """UCI engine state: board, Stockfish process, and the search worker."""

    def __init__(self) -> None:
        self.stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.board = chess.Board()
        self.board_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.session = requests.Session()  # reuses the TLS connection between moves
        self.search_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    # -- output -----------------------------------------------------------
    def send(self, line: str) -> None:
        """Write one line to the GUI (thread-safe)."""
        with self.output_lock:
            print(line, flush=True)

    # -- Stockfish --------------------------------------------------------
    def _restart_stockfish(self) -> None:
        """Try to bring Stockfish back after a crash."""
        try:
            self.stockfish.close()
        except (chess.engine.EngineError, OSError):
            pass
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except (OSError, chess.engine.EngineError) as exc:
            log(f"Could not restart Stockfish: {exc}")

    def _collect_candidates(self, board: chess.Board) -> list[Candidate]:
        """Return Stockfish's best moves, dropping ones much worse than the top."""
        analysis = self.stockfish.analyse(
            board,
            chess.engine.Limit(time=STOCKFISH_THINK_TIME),
            multipv=CANDIDATE_MOVES_COUNT,
        )
        entries = analysis if isinstance(analysis, list) else [analysis]

        scored: list[Candidate] = []
        for entry in entries:
            pv = entry.get("pv")
            score = entry.get("score")
            if not pv or score is None:
                continue
            move = pv[0]
            relative = score.relative
            scored.append(
                Candidate(
                    uci=move.uci(),
                    san=board.san(move),
                    cp=relative.score(mate_score=MATE_SCORE),
                    score=str(relative),
                    line=board.variation_san(pv[:3]),
                )
            )

        if not scored:
            return []
        best_cp = scored[0].cp  # multipv results are ordered best-first
        return [c for c in scored if c.cp >= best_cp - MAX_CANDIDATE_LOSS_CP]

    # -- Gemini -----------------------------------------------------------
    def _post_in_thread(self, payload: dict, read_timeout: float) -> queue.Queue:
        """Fire the HTTP request in a daemon thread and return a result queue.

        ``requests`` timeouts apply per phase, not in total, so the caller
        enforces the real deadline by polling this queue.
        """
        result: queue.Queue = queue.Queue(maxsize=1)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent"
        )

        def runner() -> None:
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    # Header, not query string: keeps the key out of URLs/logs.
                    headers={"x-goog-api-key": API_KEY},
                    timeout=(API_CONNECT_TIMEOUT_SECONDS, read_timeout),
                )
                response.raise_for_status()
                result.put(("ok", response.json()))
            except (requests.RequestException, ValueError) as exc:
                result.put(("err", exc))

        threading.Thread(target=runner, daemon=True, name="gemini-request").start()
        return result

    def _ask_gemini(
        self,
        board: chess.Board,
        candidates: list[Candidate],
        timeout: float,
        stop_event: threading.Event,
    ) -> tuple[str, list[str]]:
        """Ask Gemini to choose among the candidates.

        Raises:
            LLMError: with a short reason when the fallback should be used.
        """
        generation_config: dict = {"temperature": 0.4, "maxOutputTokens": MAX_OUTPUT_TOKENS}
        if GEMINI_THINKING_BUDGET is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": GEMINI_THINKING_BUDGET}

        payload = {
            "contents": [
                {"parts": [{"text": build_prompt(board, candidates, get_custom_instructions())}]}
            ],
            "generationConfig": generation_config,
        }

        result = self._post_in_thread(payload, read_timeout=timeout)
        deadline = time.monotonic() + timeout
        while True:
            if stop_event.is_set():
                raise LLMError("stopped")
            if time.monotonic() >= deadline:
                raise LLMError("timeout")
            try:
                kind, value = result.get(timeout=0.05)
                break
            except queue.Empty:
                continue

        if kind == "err":
            raise LLMError(describe_request_error(value)) from value

        text = extract_response_text(value)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise LLMError("empty response")

        chosen_uci = extract_uci_move(lines[0])
        if chosen_uci is None:
            raise LLMError("could not parse move")
        # Candidates are all legal, so membership also guarantees legality.
        if chosen_uci not in {c.uci for c in candidates}:
            raise LLMError(f"{chosen_uci} not among candidates")

        return chosen_uci, lines[1:]

    # -- Move selection ---------------------------------------------------
    def _think(
        self,
        board: chess.Board,
        params: GoParams,
        stop_event: threading.Event,
    ) -> tuple[str, list[str]]:
        """Pick a move; returns (uci, commentary lines)."""
        started = time.monotonic()

        legal = next(iter(board.legal_moves), None)
        if legal is None:
            return "0000", []  # checkmate/stalemate: UCI "null move"

        try:
            candidates = self._collect_candidates(board)
        except chess.engine.EngineError as exc:  # includes EngineTerminatedError
            log(f"Stockfish error: {exc}")
            self._restart_stockfish()
            return legal.uci(), [fallback_note("Stockfish error")]

        if not candidates:
            return legal.uci(), [fallback_note("no candidates")]

        best_uci = candidates[0].uci
        if len(candidates) == 1:
            return best_uci, [fallback_note("single reasonable move")]
        if stop_event.is_set():
            return best_uci, [fallback_note("stopped")]
        if not API_KEY:
            return best_uci, [fallback_note("GEMINI_API_KEY not set")]

        timeout = API_TIMEOUT_SECONDS
        budget = compute_move_budget(params, board.turn)
        if budget is not None:
            available = budget - (time.monotonic() - started)
            if available < MIN_API_BUDGET_SECONDS:
                return best_uci, [fallback_note("not enough time")]
            timeout = min(timeout, available)

        try:
            return self._ask_gemini(board, candidates, timeout, stop_event)
        except LLMError as exc:
            log(f"Gemini step failed: {exc}")
            return best_uci, [fallback_note(str(exc))]

    def _search_worker(
        self,
        board: chess.Board,
        params: GoParams,
        stop_event: threading.Event,
    ) -> None:
        """Run one search and always finish with a ``bestmove``."""
        try:
            move, comments = self._think(board, params, stop_event)
            if params.infinite:
                stop_event.wait()  # UCI: don't answer 'go infinite' until 'stop'
        except Exception:  # noqa: BLE001 - a GUI hangs forever without a bestmove
            traceback.print_exc(file=sys.stderr)
            legal = next(iter(board.legal_moves), None)
            move = legal.uci() if legal else "0000"
            comments = [fallback_note("internal error")]

        for comment in comments:
            # Each 'info string' must be on its own line for GUIs to show it.
            self.send(f"info string {comment}")
        self.send(f"bestmove {move}")

    def start_search(self, params: GoParams) -> None:
        """Start thinking about the current position in a worker thread."""
        self.stop_search()  # a well-behaved GUI never overlaps, but be safe
        with self.board_lock:
            board = self.board.copy()

        stop_event = threading.Event()
        self.stop_event = stop_event
        self.search_thread = threading.Thread(
            target=self._search_worker,
            args=(board, params, stop_event),
            daemon=True,
            name="search",
        )
        self.search_thread.start()

    def stop_search(self) -> None:
        """Stop any running search and wait for it to emit its bestmove."""
        thread = self.search_thread
        if thread is not None and thread.is_alive():
            self.stop_event.set()
            thread.join(timeout=5)
        self.search_thread = None

    # -- UCI command dispatch ----------------------------------------------
    def handle(self, line: str) -> bool:
        """Handle one UCI line. Returns False when the engine should exit."""
        tokens = line.split()
        if not tokens:
            return True
        command = tokens[0]

        if command == "uci":
            self.send("id name Gemini-Coach-UCI")
            self.send("id author Custom")
            self.send("uciok")
        elif command == "isready":
            self.send("readyok")
        elif command == "ucinewgame":
            self.stop_search()
            with self.board_lock:
                self.board.reset()
        elif command == "position":
            try:
                new_board = parse_position(tokens)
            except ValueError as exc:
                log(f"Ignoring invalid position command: {exc}")
            else:
                with self.board_lock:
                    self.board = new_board
        elif command == "go":
            self.start_search(parse_go(tokens))
        elif command == "stop":
            self.stop_event.set()  # worker answers immediately with its fallback
        elif command == "quit":
            return False
        return True

    def shutdown(self) -> None:
        """Release the worker, Stockfish and the HTTP session."""
        self.stop_search()
        try:
            self.stockfish.quit()
        except (chess.engine.EngineError, OSError):
            self.stockfish.close()
        self.session.close()


def main() -> None:
    """Run the UCI standard input/output event loop."""
    try:
        coach = CoachEngine()
    except FileNotFoundError:
        log(
            f"Stockfish binary not found at '{STOCKFISH_PATH}'. "
            "Install Stockfish or set the STOCKFISH_PATH environment variable."
        )
        sys.exit(1)
    except (OSError, chess.engine.EngineError) as exc:
        log(f"Could not start Stockfish at '{STOCKFISH_PATH}': {exc}")
        sys.exit(1)

    try:
        for line in iter(sys.stdin.readline, ""):  # "" means EOF (GUI closed the pipe)
            try:
                if not coach.handle(line):
                    break
            except Exception:  # noqa: BLE001 - one bad command must not kill the engine
                traceback.print_exc(file=sys.stderr)
    finally:
        coach.shutdown()


if __name__ == "__main__":
 
