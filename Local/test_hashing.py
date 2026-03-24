import unittest
import json
import os
import sys

# Add Local to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from app import hash_registry

class TestHashing(unittest.TestCase):
    def test_hash_consistency(self):
        """Test that same data produces same hash."""
        data = {"cedula": "1712345678", "num": "999", "monto": "10.00"}
        h1 = hash_registry(data)
        h2 = hash_registry(data)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_hash_uniqueness(self):
        """Test that different data produces different hashes."""
        h1 = hash_registry({"cedula": "1", "num": "999", "monto": "10.00"})
        h2 = hash_registry({"cedula": "2", "num": "999", "monto": "10.00"})
        self.assertNotEqual(h1, h2)

    def test_key_ordering_independence(self):
        """Test that key order in dict doesn't change hash."""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        self.assertEqual(hash_registry(d1), hash_registry(d2))

if __name__ == "__main__":
    unittest.main()
