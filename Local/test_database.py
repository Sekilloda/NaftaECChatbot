import os
import tempfile
import unittest
from unittest.mock import patch

import core.database as db


class TestDatabaseWrites(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = os.path.join(self._tmpdir.name, "test_chat_history.db")
        self._db_patcher = patch("core.database.DB_PATH", self._db_path)
        self._db_patcher.start()
        db.init_db()

    def tearDown(self):
        self._db_patcher.stop()
        self._tmpdir.cleanup()

    def test_save_validated_registry_is_idempotent(self):
        payload = {
            "unique_id": "dup_case_1",
            "sender_jid": "593900000003@s.whatsapp.net",
            "cedula": "1712345678",
            "banco": "Pichincha",
            "monto": "10.00",
            "fecha": "01/01/2024",
            "numero_comprobante": "999",
            "cuenta_origen": "1234567890",
        }

        first_inserted = db.save_validated_registry(payload)
        second_inserted = db.save_validated_registry(payload)

        self.assertTrue(first_inserted)
        self.assertFalse(second_inserted)


if __name__ == "__main__":
    unittest.main()
