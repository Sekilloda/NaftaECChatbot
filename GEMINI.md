# NaftaECChatbot

NaftaECChatbot is a production-ready WhatsApp-based chatbot designed to process race receipts using Optical Character Recognition (OCR) and provide intelligent RAG-based responses to runners using a streamlined Gemini 2.5/3.stack.

## Project Overview

*   **Main Entry Point:** `Local/app.py` (Flask-based webhook for WhatsApp).
*   **OCR Engine:** `Local/core/ocr.py` implements a **Vision-First** extraction pipeline using Gemini 2.5 Flash.
*   **LLM Strategy:** Optimized for the Gemini API (Paid Tier) and **Low-RAM** environments.
    *   **Response Generation:** `gemini-2.5-flash-lite`.
    *   **OCR Extraction:** `gemini-2.5-flash` (Vision-enabled).
    *   **RAG Embeddings:** `text-embedding-004` (API-based to save local RAM).
    *   **Classification & Help:** `gemini-2.5-flash-lite`.
*   **Data Storage:** SQLite database (`Local/chat_history.db`) in WAL mode. Tables:
    *   `conversations`: Full history of user/assistant interactions.
    *   `pending_confirmations`: State persistence for the OCR flow.
    *   `user_status`: Tracks if a user is in `bot` or `ayuda` (human) mode.
    *   `validated_registries`: Permanent record of processed and hashed payments.
*   **Registration Tracking:** `Local/core/registrations.py` synchronizes with the **Njuko API** in the background, supporting flexible column mapping and robust unique identification.

## Environment & Configuration

The bot uses `load_dotenv(override=True)` to ensure `Local/.env` values take absolute priority.

### Key Variables
- `GEMINI_API_KEY`: Google AI Studio API key.
- `ADMIN_PHONE`: Comma-separated list of admin numbers.
- `WEBHOOK_SECRET`: Secret token for securing the WhatsApp webhook.
- `PERSISTENT_STORAGE_PATH`: Path to the mounted persistent disk (e.g., `/var/lib/naftaec`).
- `EXTERNAL_URL`: (Optional) Public URL of the bot (e.g., `https://mybot.onrender.com`) for backup link generation.

## The OCR & Registration Pipeline

1.  **Detection:** Bot confirms if an image is a receipt.
2.  **Extraction:** Gemini 2.5 Flash analyzes the image directly.
3.  **Correction Loop:** User can update `Banco`, `Fecha`, and `Cuenta Origen`.
4.  **Runner Collection:** User specifies 1-10 runners and provides their Cédulas.
5.  **Final Review:** Full summary for final user confirmation (`CONFIRMAR` / `REINTENTAR`).
6.  **Persistence:** Validated data is hashed and stored in `validated_registries`.

## Registration Lookups (Anti-Hallucination)

To prevent the bot from inventing registrations:
*   **Unique ID Primary:** The bot is instructed to strictly request a **Cédula** for registration checks if not automatically detected by phone.
*   **Multi-Entry Support:** A single Cédula can return multiple registrations across different races.
*   **Precision Handling:** The system handles Cédulas and Phones as strict strings, avoiding float precision issues from Excel reports.

## Media & Resource Management

*   **Auto-Cleanup:** Background thread deletes files in `media/` older than **7 days**.
*   **Secure Backups:** Admins can trigger `#backup` to receive a time-limited (1-hour) secure link to a ZIP containing the database and current registry.

## Security & Admin Features

*   **Webhook Authentication:** Requires `X-Webhook-Secret` or `secret` param.
*   **Admin Commands:**
    *   `#resuelto` / `#resolver`: Resets a user's status from `ayuda` back to `bot`.
    *   `#backup`: Generates and sends a secure database backup link.
*   **Help Escalation:** AI automatically detects help requests and offers to switch to `ayuda` mode.

## Deployment on Render

Deployed via **Docker** with a **Persistent Disk** for the SQLite database and media.

### Render Configuration
1.  **Service Type:** Web Service (Docker).
2.  **Persistent Disk:** Mounted at `/var/lib/naftaec`.
3.  **Environment Variable:** `PERSISTENT_STORAGE_PATH=/var/lib/naftaec`.

## Testing & Validation
*   **Registration Flow:** `python Local/test_registration_flow.py` (Validates dynamic lookups and anti-hallucination).
*   **Admin Backup:** `python Local/test_admin_backup.py` (Verifies secure backup generation and tokens).
*   **OCR Accuracy:** `python Local/test_ocr_accuracy.py`.
*   **State Machine:** `python Local/test_state_machine.py`.
