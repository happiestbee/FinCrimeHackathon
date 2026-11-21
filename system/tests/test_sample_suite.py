from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import OpenRouterClient
from src.main import run_pipeline


SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "sample_data"


class TestSampleSuite(unittest.TestCase):
    def test_synthetic_sample_suite_all_cases_and_determinism(self) -> None:
        expected_path = SAMPLE_DATA_DIR / "expected_labels.jsonl"
        expected = {}
        with expected_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                expected[row["hit_id"]] = row

        self.assertEqual(len(expected), 20)

        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / "run1.jsonl"
            out2 = Path(tmpdir) / "run2.jsonl"
            offline_llm = OpenRouterClient(api_key="")
            offline_llm.api_key = ""
            run_pipeline(str(SAMPLE_DATA_DIR), str(out1), llm_client=offline_llm)
            run_pipeline(str(SAMPLE_DATA_DIR), str(out2), llm_client=offline_llm)

            raw1 = out1.read_text(encoding="utf-8")
            raw2 = out2.read_text(encoding="utf-8")
            self.assertEqual(raw1, raw2, "Pipeline output was not byte-identical across runs")

            actual = {}
            for line in raw1.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                actual[row["hit_id"]] = row

            self.assertEqual(set(actual.keys()), set(expected.keys()))

            for hit_id, exp in sorted(expected.items()):
                got = actual[hit_id]
                with self.subTest(hit_id=hit_id):
                    self.assertEqual(got["decision"], exp["decision"])
                    self.assertIn(got["reason_code"], exp["reason_codes"])
                    self.assertIn(got["customer_record_id"], exp["customer_record_ids"])
                    self.assertTrue(got["reason"].strip())
                    self.assertIsInstance(got["evidence"], list)
                    self.assertGreater(len(got["evidence"]), 0)


if __name__ == "__main__":
    unittest.main()
