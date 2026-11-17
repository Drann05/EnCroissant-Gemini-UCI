# Gemini Coach UCI (Linux only)

A custom **UCI (Universal Chess Interface)** chess engine wrapper written in Python that bridges **Stockfish** with the **Google Gemini API**.

Designed for chess GUIs like **En Croissant**, **Arena**, or **Lucas Chess**, this tool allows you to play against or analyze with an LLM that follows custom natural-language instructions (e.g., playing specific openings, simulating a target Elo, or avoiding queen trades) while explaining its strategic and tactical reasoning in real time.

## How It Works

Pure LLMs struggle with chess calculation and frequently hallucinate illegal moves. This wrapper solves that with a hybrid architecture:

1. **Tactical Grounding (Stockfish):** On every turn, local Stockfish evaluates the current FEN position (`MultiPV=6`) to generate a shortlist of top legal candidate moves alongside their centipawn evaluations and principal variations.
2. **Strategic Selection & Coaching (Gemini API):** The script sends the board state, Stockfish's candidate lines, and your custom prompt instructions to `gemini-3.5-flash-lite`. Gemini selects the move that best fits your requested playstyle and generates a multi-sentence pedagogical explanation.
3. **UCI Integration:** The explanation is streamed back to the chess GUI via standard `info string` outputs, followed by the validated `bestmove`.

## Features

- **Dynamic Playstyle Customization:** Reads instructions from `~/.config/gemini_chess_prompt.txt` on every move—allowing you to change the bot's behavior, opening repertoire, or difficulty mid-game without restarting your GUI.
- **Real-Time Move Explanations:** Outputs detailed tactical and positional commentary (threats, key squares, piece coordination) directly inside your GUI's engine analysis panel.
- **Zero Illegal Moves:** Enforces move legality via `python-chess` validation and automatically falls back to Stockfish's top choice if the API times out or errors out.
- **IPv4 Socket Enforcement:** Includes an `urllib3` IPv4 monkey-patch to prevent dual-stack connection timeouts on Linux setups.

## Prerequisites

- **Python 3.10+**
- **Stockfish** installed locally (default path: `/usr/bin/stockfish`)
- Required Python packages:
  ```bash
  pip install chess requests urllib3
  ```
- A valid **Google Gemini API Key**

## Setup & Installation

1. **Save the script** to your local binary path and make it executable:
   ```bash
   chmod +x ~/.local/bin/gemini_uci.py
   ```

2. **Set your Gemini API Key** as an environment variable:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
   *(If launching your chess GUI from a desktop environment/window manager, make sure `GEMINI_API_KEY` is exported in your session environment).*

3. **Create your custom instructions file:**
   ```bash
   echo "Play like a 1000 Elo club player. Prefer open games, avoid early queen trades, and explain your strategic plan clearly." > ~/.config/gemini_chess_prompt.txt
   ```

4. **Test via CLI:**
   Verify that the script responds properly to standard UCI commands:
   ```bash
   printf "uci\nisready\nposition startpos moves e2e4 e7e5\ngo\n" | ~/.local/bin/gemini_uci.py
   ```

## Usage in En Croissant

1. Open **En Croissant** and navigate to **Engines** -> **Add New** -> **Local Engine**.
2. Select `~/.local/bin/gemini_uci.py`.
3. Start a game via **Play** -> **Play vs Computer** and select **Gemini-Coach-UCI** as your opponent.
4. Keep the **Analysis / Engine** tab open during the match to read Gemini's move-by-move explanations.
