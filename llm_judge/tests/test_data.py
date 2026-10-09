import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.config import Config
from pipeline.data import download_metadata


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.cfg = Config(gt_csv=root / "gt.csv", crowd_csv=root / "crowd_labels.csv")
        self.content = {"gt.csv": b"label,score\nhttps://example.com/face.jpg,25\n",
                        "crowd_labels.csv": b"left,right,label,performer\na,b,a,0\n"}
        self.manifest = {name: (hashlib.sha256(value).hexdigest(), set(value.decode().splitlines()[0].split(",")))
                         for name, value in self.content.items()}

    def response(self, url, **kwargs):
        result = MagicMock()
        result.__enter__.return_value = result
        result.iter_content.return_value = [self.content[url.rsplit("/", 1)[-1]]]
        return result

    def test_download_and_idempotent_rerun(self):
        with patch("pipeline.data.FILES", self.manifest), patch("pipeline.data.requests.get", side_effect=self.response) as get:
            download_metadata(self.cfg)
            download_metadata(self.cfg)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(self.cfg.gt_csv.read_bytes(), self.content["gt.csv"])
        self.assertFalse(list(self.cfg.gt_csv.parent.glob("*.part")))

    def test_bad_download_not_promoted(self):
        response = self.response("gt.csv")
        response.iter_content.return_value = [b"invalid"]
        with patch("pipeline.data.FILES", self.manifest), patch("pipeline.data.requests.get", return_value=response):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                download_metadata(self.cfg)
        self.assertFalse(self.cfg.gt_csv.exists())
        self.assertFalse(list(self.cfg.gt_csv.parent.glob("*.part")))

    def test_existing_custom_file_not_overwritten(self):
        self.cfg.gt_csv.write_bytes(b"custom-data")
        with patch("pipeline.data.FILES", self.manifest), patch("pipeline.data.requests.get") as get:
            with self.assertRaises(ValueError):
                download_metadata(self.cfg)
        get.assert_not_called()
        self.assertEqual(self.cfg.gt_csv.read_bytes(), b"custom-data")

    def test_network_failure_removes_partial_file(self):
        response = self.response("gt.csv")
        response.iter_content.side_effect = RuntimeError("connection interrupted")
        with patch("pipeline.data.requests.get", return_value=response):
            with self.assertRaises(RuntimeError):
                download_metadata(self.cfg)
        self.assertFalse(self.cfg.gt_csv.exists())
        self.assertFalse(list(self.cfg.gt_csv.parent.glob("*.part")))


if __name__ == "__main__":
    unittest.main()
