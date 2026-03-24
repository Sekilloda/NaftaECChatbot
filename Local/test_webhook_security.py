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


if __name__ == "__main__":
    unittest.main()
