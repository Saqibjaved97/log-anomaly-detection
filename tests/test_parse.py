import unittest

from loganomaly.parse import parse_line, template


class TemplateTests(unittest.TestCase):
    def test_masks_variable_parts(self):
        self.assertEqual(template("User 4821 logged in from 10.0.3.17"), "User <NUM> logged in from <IP>")
        self.assertEqual(template("Retrying transaction TX-123456 attempt 2"), "Retrying transaction <ID> attempt <NUM>")

    def test_same_event_same_template(self):
        a = template("GET /v1/payments/8812 200 took 45ms")
        b = template("GET /v1/payments/77 200 took 1203ms")
        self.assertEqual(a, b)


class ParseLineTests(unittest.TestCase):
    def test_parses_fields_and_latency(self):
        rec = parse_line("2026-10-01T00:00:09.197Z INFO api POST /v1/payments 201 took 43ms")
        self.assertEqual(rec["level"], "INFO")
        self.assertEqual(rec["service"], "api")
        self.assertEqual(rec["latency_ms"], 43.0)
        self.assertEqual(rec["template"], "api: POST /v<NUM>/payments <NUM> took <NUM>ms")

    def test_no_latency(self):
        rec = parse_line("2026-10-01T00:00:09.197Z WARN auth Failed login for user 1234 from 1.2.3.4")
        self.assertIsNone(rec["latency_ms"])

    def test_rejects_garbage(self):
        self.assertIsNone(parse_line("not a log line"))
        self.assertIsNone(parse_line("yesterday INFO api hello"))


if __name__ == "__main__":
    unittest.main()
