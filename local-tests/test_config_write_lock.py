"""Tests of the config write-lock port (upstream fix #147939), run in the image build.

The patch adds a reentrant cross-process ``utils.files._yaml_write_lock`` (``hermes_cli.auth._file_lock``
on ``<file>.lock``) around every round-trip YAML load-mutate-dump, adds
``atomic_roundtrip_yaml_updates`` (several dotted keys in one locked cycle), makes
``FileBackend.write_path`` publish a grouped set/unset as one write, and holds the lock across
``persist_model_selection``'s read -> merge -> replace. The tests show:
 - 2 processes that each do 50 read-modify-write updates of different keys of 1 YAML file lose
   no key: all 100 keys are present at the end;
 - a grouped set/unset through ``FileBackend.write_path`` applies all keys in one write;
 - ``drop_other_access`` behavior is unchanged: world bits are stripped, group bits stay.
"""
import multiprocessing
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hermes_yaml as yaml

PER_PROCESS = 50


def _writer(home, config_path, prefix, start):
    os.environ["HERMES_HOME"] = home
    from hermes_cli.config_backend import Changes, FileBackend
    start.wait(30)
    for i in range(PER_PROCESS):
        FileBackend.write_path(Path(config_path), Changes(set={f"{prefix}.k{i}": i}))


class _TempHome(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        self._env = patch.dict(os.environ, {"HERMES_HOME": self.home})
        self._env.start()
        self.config = Path(self.home) / "config.yaml"
        self.config.write_text("keep: 1\n", encoding="utf-8")

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def read(self):
        return yaml.safe_load(self.config.read_text(encoding="utf-8"))


class ConcurrentWritersTest(_TempHome):
    def test_two_processes_lose_no_key(self):
        ctx = multiprocessing.get_context("spawn")
        start = ctx.Event()
        procs = [ctx.Process(target=_writer, args=(self.home, str(self.config), p, start))
                 for p in ("a", "b")]
        for p in procs:
            p.start()
        start.set()
        for p in procs:
            p.join(300)
        self.assertEqual([p.exitcode for p in procs], [0, 0])
        data = self.read()
        self.assertEqual(data["keep"], 1)
        for prefix in ("a", "b"):
            self.assertEqual(data[prefix], {f"k{i}": i for i in range(PER_PROCESS)})


class GroupedWriteTest(_TempHome):
    def test_grouped_set_and_unset_in_one_write(self):
        from hermes_cli.config_backend import Changes, FileBackend
        import utils.files as files

        self.config.write_text("keep: 1\nmodel:\n  default: old\n  api_key: x\n", encoding="utf-8")
        real = files._roundtrip_dump
        dumps = []

        def counting_dump(*args, **kwargs):
            dumps.append(args[0])
            return real(*args, **kwargs)

        with patch.object(files, "_roundtrip_dump", counting_dump):
            FileBackend.write_path(self.config, Changes(
                set={"model.default": "new", "model.provider": "p"}, unset=("model.api_key",)))
        self.assertEqual(len(dumps), 1)
        self.assertEqual(self.read(), {"keep": 1, "model": {"default": "new", "provider": "p"}})

    def test_updates_helper_applies_all_keys(self):
        from utils.files import atomic_roundtrip_yaml_updates

        atomic_roundtrip_yaml_updates(self.config, {"a.b": 2, "c": "d", "keep": None})
        self.assertEqual(self.read(), {"a": {"b": 2}, "c": "d"})


class DropOtherAccessTest(_TempHome):
    def _write_with_mode(self, mode, changes):
        from hermes_cli.config_backend import FileBackend
        os.chmod(self.config, mode)
        with patch("hermes_constants._container_or_chmod_skipped", return_value=False), \
             patch("hermes_constants.get_managed_system", return_value=None):
            FileBackend.write_path(self.config, changes)
        return stat.S_IMODE(self.config.stat().st_mode)

    def test_world_bits_stripped_group_bits_kept(self):
        from hermes_cli.config_backend import Changes
        self.assertEqual(self._write_with_mode(0o666, Changes(set={"x": 1, "y": 2})), 0o660)
        self.assertEqual(self._write_with_mode(0o640, Changes(set={"z": 3})), 0o640)


if __name__ == "__main__":
    unittest.main()
