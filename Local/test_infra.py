import unittest
import os
import sys
import base64
from unittest.mock import patch, MagicMock
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256

# Add Local to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from core.whatsapp import decrypt_and_save_media, normalize_phone, download_media

class TestInfra(unittest.TestCase):
    def test_normalize_phone(self):
        self.assertEqual(normalize_phone("+1 234-567 890"), "1234567890")
        self.assertEqual(normalize_phone("593991234567@s.whatsapp.net"), "593991234567")

    @patch('requests.get')
    def test_download_media(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"fake_media_data"
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        data = download_media("http://example.com/media")
        self.assertEqual(data, b"fake_media_data")
        mock_get.assert_called_once()

    def test_decrypt_and_save_media_logic(self):
        """Verify decryption logic by simulating an encrypted buffer."""
        # Generate a dummy 32-byte key
        dummy_key = os.urandom(32)
        dummy_key_b64 = base64.b64encode(dummy_key).decode('utf-8')
        
        # Manually derive keys as core/whatsapp.py does
        app_info = b'WhatsApp Image Keys'
        expanded_keys = HKDF(master=dummy_key, key_len=48, salt=b'', hashmod=SHA256, context=app_info, num_keys=1)
        iv = expanded_keys[0:16]
        cipher_key = expanded_keys[16:48]
        
        # Create some plain data and pad it
        plain_data = b"Hello, WhatsApp OCR!"
        padded_data = pad(plain_data, AES.block_size, style='pkcs7')
        
        # Encrypt
        cipher = AES.new(cipher_key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(padded_data)
        
        # WhatsApp adds a 10-byte HMAC-like suffix (ignored by our logic but needed for length check)
        full_encrypted_data = ciphertext + os.urandom(10)
        
        test_output_path = os.path.join(BASE_DIR, "test_decrypted.jpg")
        
        try:
            # Run decryption
            success = decrypt_and_save_media(dummy_key_b64, full_encrypted_data, test_output_path, "image")
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(test_output_path))
            
            with open(test_output_path, 'rb') as f:
                saved_data = f.read()
            
            self.assertEqual(saved_data, plain_data)
        finally:
            if os.path.exists(test_output_path):
                os.remove(test_output_path)

if __name__ == "__main__":
    unittest.main()
