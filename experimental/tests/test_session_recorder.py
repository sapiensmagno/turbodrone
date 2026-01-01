import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.session_recorder import SessionRecorder, build_default_meta, measure_icmp_ping_rtt_ms


class _FakeCompleted:
    def __init__(self, *, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


class TestMeasureIcmpPingRttMs(unittest.TestCase):
    def test_parses_avg_from_min_avg_max_summary(self) -> None:
        out = "rtt min/avg/max/mdev = 9.100/12.300/15.200/1.100 ms\n"
        with mock.patch("subprocess.run", return_value=_FakeCompleted(stdout=out, stderr="")) as m:
            rtt = measure_icmp_ping_rtt_ms("example")

        self.assertIsNotNone(rtt)
        assert rtt is not None
        self.assertAlmostEqual(rtt, 12.3, places=3)
        self.assertTrue(m.called)

    def test_parses_single_time_value(self) -> None:
        out = "64 bytes from 1.2.3.4: icmp_seq=1 ttl=64 time=7.89 ms\n"
        with mock.patch("subprocess.run", return_value=_FakeCompleted(stdout=out, stderr="")):
            rtt = measure_icmp_ping_rtt_ms("example", count=1)

        self.assertIsNotNone(rtt)
        assert rtt is not None
        self.assertAlmostEqual(rtt, 7.89, places=3)

    def test_returns_none_when_no_match(self) -> None:
        out = "ping: unknown host\n"
        with mock.patch("subprocess.run", return_value=_FakeCompleted(stdout=out, stderr="")):
            rtt = measure_icmp_ping_rtt_ms("example")

        self.assertIsNone(rtt)


class TestSessionRecorder(unittest.TestCase):
    def test_writes_meta_and_samples_and_updates_meta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            r = SessionRecorder(base_dir=base)

            meta = build_default_meta(cfg={"x": 1}, net_rtt_ms=12.0, mode="test")
            sd = r.start(meta=meta)

            self.assertTrue(sd.exists())
            self.assertTrue((sd / "meta.json").exists())
            self.assertTrue((sd / "samples.jsonl").exists())

            raw = (sd / "meta.json").read_text(encoding="utf-8")
            meta_loaded = json.loads(raw)
            self.assertEqual(meta_loaded["schema_version"], 1)
            self.assertEqual(meta_loaded["mode"], "test")
            self.assertEqual(meta_loaded["net_rtt_ms"], 12.0)

            r.update_meta({"hello": "world"})
            meta_loaded2 = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta_loaded2["hello"], "world")

            @dataclass
            class _D:
                x: int

            r.write_sample({"a": 1, "d": _D(x=2)})
            lines = (sd / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 1)
            sample = json.loads(lines[-1])
            self.assertEqual(sample["a"], 1)
            self.assertEqual(sample["d"]["x"], 2)

            r.close()


if __name__ == "__main__":
    unittest.main()
