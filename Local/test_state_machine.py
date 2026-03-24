import unittest
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add Local to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from app import app
import core.database as db

class TestOCRStateMachine(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        # Mock database/runtime helpers to avoid side effects.
        self.bg_patcher = patch('app.ensure_background_services')
        self.bg_patcher.start()

        self.user_status_patcher = patch('app.get_user_status', return_value='bot')
        self.user_status_patcher.start()

        self.save_message_patcher = patch('app.save_message')
        self.save_message_patcher.start()

        self.db_patcher = patch('app.get_pending_confirmation')
        self.mock_get_pending = self.db_patcher.start()
        
        self.save_patcher = patch('app.save_pending_confirmation')
        self.mock_save_pending = self.save_patcher.start()
        
        self.clear_patcher = patch('app.clear_pending_confirmation')
        self.mock_clear_pending = self.clear_patcher.start()
        
        self.msg_patcher = patch('app.send_whatsapp_message')
        self.mock_send_msg = self.msg_patcher.start()
        
        self.auth_patcher = patch('app.is_authorized_webhook', return_value=True)
        self.auth_patcher.start()

    def tearDown(self):
        patch.stopall()

    def simulate_message(self, sender, text):
        payload = {
            "event": "messages.upsert",
            "data": {
                "messages": [{
                    "key": {"remoteJid": sender, "id": "msg123", "fromMe": False},
                    "message": {"conversation": text},
                    "pushName": "Test User"
                }]
            }
        }
        return self.client.post('/webhook', json=payload)

    def test_flow_block_monto_edit(self):
        """Test that editing Monto is blocked."""
        sender = "12345@s.whatsapp.net"
        # Setup pending state: OCR_EDIT_MODE
        self.mock_get_pending.return_value = {
            "sender_jid": sender,
            "state": "OCR_EDIT_MODE",
            "metadata": {"ocr_data": {"banco": "Pichincha", "monto": "10.00", "numero_comprobante": "999"}}
        }
        
        # User tries to change monto
        self.simulate_message(sender, "Monto: 50.00")
        
        # Verify block message was sent
        args, _ = self.mock_send_msg.call_args
        self.assertIn("Monto y el Número de comprobante no pueden ser editados", args[1])
        # Verify state was NOT changed to something else incorrectly
        # (save_pending_confirmation should not have been called with a new state)
        for call in self.mock_save_pending.call_args_list:
            self.assertNotEqual(call[0][1]['state'], 'OCR_AWAITING_RUNNER_COUNT')

    def test_flow_complete_registration(self):
        """Test full flow: Correcto -> Count -> Cedulas -> Final Confirmation."""
        sender = "12345@s.whatsapp.net"
        
        # 1. User says "Correcto"
        self.mock_get_pending.return_value = {
            "sender_jid": sender,
            "state": "OCR_EDIT_MODE",
            "metadata": {"ocr_data": {"banco": "Pichincha", "monto": "10.00", "numero_comprobante": "999", "fecha": "01/01/2024", "cuenta_origen": "123"}}
        }
        self.simulate_message(sender, "Correcto")
        
        # Verify state transition
        self.mock_save_pending.assert_called()
        last_state = self.mock_save_pending.call_args[0][1]['state']
        self.assertEqual(last_state, "OCR_AWAITING_RUNNER_COUNT")

        # 2. User says "2" runners
        self.mock_get_pending.return_value = {
            "sender_jid": sender,
            "state": "OCR_AWAITING_RUNNER_COUNT",
            "metadata": {"ocr_data": {"banco": "Pichincha", "monto": "10.00", "numero_comprobante": "999"}}
        }
        self.simulate_message(sender, "2")
        self.assertEqual(self.mock_save_pending.call_args[0][1]['state'], "OCR_AWAITING_CEDULAS")
        self.assertEqual(self.mock_save_pending.call_args[0][1]['metadata']['runner_count'], 2)

        # 3. User sends first cedula
        self.mock_get_pending.return_value = {
            "sender_jid": sender,
            "state": "OCR_AWAITING_CEDULAS",
            "metadata": {"runner_count": 2, "cedulas_collected": [], "ocr_data": {}}
        }
        self.simulate_message(sender, "1712345678")
        self.assertEqual(len(self.mock_save_pending.call_args[0][1]['metadata']['cedulas_collected']), 1)
        self.assertEqual(self.mock_save_pending.call_args[0][1]['state'], "OCR_AWAITING_CEDULAS")

        # 4. User sends second cedula -> Final Summary
        self.mock_get_pending.return_value = {
            "sender_jid": sender,
            "state": "OCR_AWAITING_CEDULAS",
            "metadata": {"runner_count": 2, "cedulas_collected": ["1712345678"], "ocr_data": {"banco": "Pichincha"}}
        }
        self.simulate_message(sender, "1787654321")
        self.assertEqual(self.mock_save_pending.call_args[0][1]['state'], "OCR_FINAL_CONFIRMATION")
        
        # Verify summary message
        args, _ = self.mock_send_msg.call_args
        self.assertIn("RESUMEN FINAL", args[1])
        self.assertIn("1712345678", args[1])
        self.assertIn("1787654321", args[1])

    @patch('app.save_validated_registry')
    def test_final_confirmation(self, mock_save_reg):
        """Test final CONFIRMAR command."""
        sender = "12345@s.whatsapp.net"
        self.mock_get_pending.return_value = {
            "sender_jid": sender,
            "state": "OCR_FINAL_CONFIRMATION",
            "metadata": {
                "runner_count": 1, 
                "cedulas_collected": ["1712345678"], 
                "ocr_data": {"banco": "Pichincha", "monto": "10.00", "numero_comprobante": "999", "fecha": "01/01/2024", "cuenta_origen": "123"}
            }
        }
        
        self.simulate_message(sender, "CONFIRMAR")
        
        # Verify saved to DB
        self.assertTrue(mock_save_reg.called)
        # Verify success message
        args, _ = self.mock_send_msg.call_args
        self.assertIn("Comprobante registrado exitosamente", args[1])
        # Verify pending cleared
        self.mock_clear_pending.assert_called_with(sender)

if __name__ == "__main__":
    unittest.main()
