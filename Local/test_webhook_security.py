import unittest
import os
import sys
from unittest.mock import patch

# Add Local to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from app import app
import app as app_module


class TestWebhookSecurity(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_webhook_rejects_when_secret_missing_by_default(self):
        payload = {"event": "ping", "data": {}}
        with patch("app.ensure_background_services"), patch.object(app_module, "WEBHOOK_SECRET", ""), patch.object(app_module, "ALLOW_UNAUTHENTICATED_WEBHOOK", False):
            res = self.client.post("/webhook", json=payload)
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.get_json().get("status"), "unauthorized")

    def test_webhook_can_be_opened_only_with_explicit_opt_in(self):
        payload = {"event": "ping", "data": {}}
        with patch("app.ensure_background_services"), patch.object(app_module, "WEBHOOK_SECRET", ""), patch.object(app_module, "ALLOW_UNAUTHENTICATED_WEBHOOK", True):
            res = self.client.post("/webhook", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("status"), "ignored")

    def test_admin_sender_requires_exact_number_match(self):
        with patch.object(app_module, "ADMIN_PHONE", "+593991234567"):
            self.assertTrue(app_module.is_admin_sender("593991234567@s.whatsapp.net"))
            self.assertFalse(app_module.is_admin_sender("991234567@s.whatsapp.net"))
            self.assertFalse(app_module.is_admin_sender("39912345@s.whatsapp.net"))

    def test_admin_sender_supports_multiple_numbers(self):
        with patch.object(app_module, "ADMIN_PHONE", "+593991234567, +593998887777"):
            self.assertTrue(app_module.is_admin_sender("593998887777@s.whatsapp.net"))
            self.assertFalse(app_module.is_admin_sender("593991234568@s.whatsapp.net"))

    def test_webhook_processes_all_messages_in_batch(self):
        payload = {
            "event": "messages.upsert",
            "data": {
                "messages": [
                    {
                        "key": {"remoteJid": "593900000001@s.whatsapp.net", "id": "msg-1", "fromMe": False},
                        "message": {"conversation": "primero"},
                    },
                    {
                        "key": {"remoteJid": "593900000001@s.whatsapp.net", "id": "msg-2", "fromMe": False},
                        "message": {"conversation": "segundo"},
                    },
                ]
            },
        }
        with patch("app.ensure_background_services"), \
             patch("app.is_authorized_webhook", return_value=True), \
             patch("app.get_user_status", return_value="bot"), \
             patch("app.get_pending_confirmation", return_value=None), \
             patch("app.responder", return_value="ok"), \
             patch("app.send_whatsapp_message", return_value=True) as mock_send, \
             patch("app.save_message") as mock_save:
            res = self.client.post("/webhook", json=payload)

        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("status"), "batch_processed")
        self.assertEqual(len(body.get("results", [])), 2)
        self.assertEqual(mock_send.call_count, 2)

        user_messages = [call[0][2] for call in mock_save.call_args_list if call[0][1] == "user"]
        self.assertEqual(user_messages, ["primero", "segundo"])

    def test_media_message_id_path_is_sanitized(self):
        payload = {
            "event": "messages.upsert",
            "data": {
                "messages": [
                    {
                        "key": {"remoteJid": "593900000002@s.whatsapp.net", "id": "../../outside", "fromMe": False},
                        "message": {
                            "imageMessage": {
                                "mediaKey": "ZmFrZW1lZGlha2V5",
                                "url": "https://example.com/media",
                            }
                        },
                    }
                ]
            },
        }
        with patch("app.ensure_background_services"), \
             patch("app.is_authorized_webhook", return_value=True), \
             patch("app.get_user_status", return_value="bot"), \
             patch("app.get_pending_confirmation", return_value=None), \
             patch("app.download_media", return_value=b"encrypted"), \
             patch("app.decrypt_and_save_media", return_value=True), \
             patch("app.send_whatsapp_message", return_value=True), \
             patch("app.save_pending_confirmation") as mock_save_pending:
            res = self.client.post("/webhook", json=payload)

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("status"), "media_received_awaiting_confirmation")

        saved_payload = mock_save_pending.call_args[0][1]
        output_path = os.path.abspath(saved_payload["output_path"])
        media_root = os.path.abspath(app_module.MEDIA_DIR)

        self.assertTrue(output_path.startswith(media_root + os.sep))
        self.assertFalse(os.path.relpath(output_path, media_root).startswith(".."))
        self.assertNotIn("/", saved_payload["original_filename"])
        self.assertNotIn("\\", saved_payload["original_filename"])

    def test_admin_backup_command_succeeds(self):
        payload = {
            "event": "messages.upsert",
            "data": {
                "messages": [
                    {
                        "key": {"remoteJid": "593991234567@s.whatsapp.net", "id": "backup-msg", "fromMe": False},
                        "message": {"conversation": "#backup"},
                    }
                ]
            },
        }

        with patch("app.ensure_background_services"), \
             patch("app.is_authorized_webhook", return_value=True), \
             patch.object(app_module, "ADMIN_PHONE", "593991234567"), \
             patch("app.send_whatsapp_document", return_value=True):
            res = self.client.post("/webhook", json=payload)

        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("status"), "backup_sent")
        token = body.get("token")
        self.assertIn(token, app_module.BACKUP_TOKENS)

        file_path = app_module.BACKUP_TOKENS[token]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
        app_module.BACKUP_TOKENS.pop(token, None)


if __name__ == "__main__":
    unittest.main()
