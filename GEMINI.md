# NaftaECChatbot

NaftaECChatbot is a production-ready WhatsApp-based chatbot designed to process race receipts using Optical Character Recognition (OCR) and provide intelligent RAG-based responses to runners using a streamlined Gemini 2.5/3.x stack.

## Project Overview

*   **Main Entry Point:** `Local/app.py` (Flask-based webhook for WhatsApp).
*   **OCR Engine:** `Local/core/ocr.py` implements a **Vision-First** extraction pipeline using Gemini 2.5 Flash (sending images directly for high accuracy).
*   **LLM Strategy:** Optimized for the Gemini API (Paid Tier) and **Low-RAM** environments.
    *   **Response Generation:** `gemini-2.5-flash-lite`.
    *   **OCR Extraction:** `gemini-2.5-flash` (Vision-enabled).
    *   **RAG Embeddings:** `text-embedding-004` (API-based to save local RAM).
    *   **Classification & Help:** `gemini-2.5-flash-lite`.
*   **Data Storage:** SQLite database (`Local/chat_history.db`) in WAL mode. Tables:
    *   `conversations`: Full history of user/assistant interactions.
    *   `pending_confirmations`: State persistence for the OCR flow (including a `metadata` JSON column).
    *   `user_status`: Tracks if a user is in `bot` or `ayuda` (human) mode.
    *   `validated_registries`: Permanent record of processed and hashed payments.
*   **Registration Tracking:** `Local/core/registrations.py` synchronizes with the **Njuko API** in the background, supporting flexible column mapping and fuzzy name search.

## Environment & Configuration

The bot uses a robust environment loading mechanism (`load_dotenv(override=True)`) to ensure that values in the local `Local/.env` file (like the Paid Tier Gemini API key) take absolute priority over system-level environment variables.

### Key Variables
- `GEMINI_API_KEY`: Google AI Studio API key (Paid tier recommended for benchmarks).
- `ADMIN_PHONE`: Comma-separated list of admin numbers for human help escalation.
- `WEBHOOK_SECRET`: Secret token for securing the WhatsApp webhook.
- `PERSISTENT_STORAGE_PATH`: (Production) Path to the mounted persistent disk (e.g., `/var/lib/naftaec`).

## The OCR & Registration Pipeline (Vision-First)

To ensure maximum accuracy and security, the receipt processing flow follows a deterministic state machine:
1.  **Detection:** User sends an image; bot confirms if it's a receipt.
2.  **Extraction:** Gemini 2.5 Flash analyzes the image directly.
    - **Normalization:** `monto` is standardized to 2 decimals (e.g., `20.00`).
    - **Masking Detection:** Fields like `cuenta_origen` are cleared (nullified) if they contain masking characters (e.g., `****`) to avoid storing partial/obfuscated data.
3.  **Correction Loop (`OCR_EDIT_MODE`):** User can update `Banco`, `Fecha`, and `Cuenta Origen`. Edits to `Monto` and `Comprobante` are **blocked** for security.
4.  **Runner Collection:** User specifies the number of runners (1-10) and provides their ID numbers (Cédulas).
5.  **Final Review:** A full summary is displayed for final user confirmation (`CONFIRMAR` / `REINTENTAR`).
6.  **Persistence:** Validated data is hashed (SHA-256) and stored in `validated_registries`.

## Media Management

The bot stores decrypted receipts in the `media/` folder within the persistent storage path. To manage disk space efficiently:
*   **Auto-Cleanup:** A background thread automatically deletes media files older than **7 days**. This provides a sufficient window for debugging while preventing disk overflow.

## Security & Admin Features

*   **Webhook Authentication:** Webhooks require a secret provided via `X-Webhook-Secret` or `secret` param.
*   **Admin Commands:** `#resolver <phone>` resets a user's status from `ayuda` back to `bot`.
*   **Help Escalation:** AI automatically detects help requests and offers to switch to `ayuda` mode.

## Deployment on Render

The bot is deployed using **Docker** to ensure all system dependencies (Tesseract, OpenCV) are present. It requires a **Persistent Disk** to maintain the SQLite database and decrypted media between restarts.

### Render Configuration
1.  **Service Type:** Web Service (Docker).
2.  **Persistent Disk:**
    *   **Mount Path:** `/var/lib/naftaec`
    *   **Environment Variable:** `PERSISTENT_STORAGE_PATH=/var/lib/naftaec`
3.  **Environment Variables:**
    *   `GEMINI_API_KEY`, `WASENDER_API_TOKEN`, `WEBHOOK_SECRET`.
    *   `ADMIN_PHONE` (e.g., `+593991234567, +593998887777`).
    *   `PORT=5001`.
4.  **Gunicorn Setup:** The `Dockerfile` uses 1 worker with multiple threads to ensure SQLite thread-safety.

## Testing & Validation
*   **OCR Accuracy Benchmark:** `python Local/test_ocr_accuracy.py` (Benchmarked at 100% for Fecha/Comprobante using Vision-First).
*   **State Machine Test:** `python Local/test_state_machine.py` (Validates deterministic flow).
*   **Webhook Security Test:** `python Local/test_webhook_security.py` (Verifies auth logic).
*   **Persona Simulation:** `python Local/simulation/run_simulation.py` (Scored AI conversations).
