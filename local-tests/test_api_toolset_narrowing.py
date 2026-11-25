"""Tests of the per-request toolset narrowing port (fork finding #21) in the API server.

The patch lets an API request name ``enabled_toolsets``. ``_request_agent_overrides`` parses the
field into ``enabled_toolsets_override`` and ``APIServerAdapter._create_agent`` intersects it with
the toolsets that the profile config (``platform_toolsets.api_server``) grants. The request names
resolve like a saved config list (a composite such as ``hermes-api-server`` expands), so a request
can only remove toolsets. The tests show:
 - a request that adds ``terminal`` to a profile without it gets no ``terminal``;
 - a request naming the profile composite plus more gets exactly the configured set;
 - an empty list or no field keeps the configured set;
 - a subset request gets the subset;
 - the tool names that the effective toolsets resolve to are always a subset of the configured ones.
The AIAgent constructor and the provider runtime lookup are patched: they need provider credentials,
and the tests read only the keyword arguments that ``_create_agent`` gives to AIAgent.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HOME = tempfile.mkdtemp(prefix="hermes-home-")
_ENV = patch.dict(os.environ, {"HERMES_HOME": _HOME})
_CONFIG = os.path.join(_HOME, "config.yaml")


def setUpModule():
    # The Hermes imports touch HERMES_HOME at import time, so they run after the patch starts.
    global gateway_run, PlatformConfig, APIServerAdapter, _request_agent_overrides
    global _get_platform_tools, _select_tool_names
    _ENV.start()
    import gateway.run as gateway_run
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, _request_agent_overrides
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import _select_tool_names


def tearDownModule():
    _ENV.stop()


def _write_profile(toolsets):
    with open(_CONFIG, "w") as fh:
        fh.write("model:\n  default: fixture-model\nplatform_toolsets:\n  api_server:\n")
        for name in toolsets:
            fh.write(f"    - {name}\n")


def _effective_toolsets(profile, body):
    """Run the real parse + _create_agent path; return the enabled_toolsets given to AIAgent."""
    _write_profile(profile)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    overrides = _request_agent_overrides(body, virtual_model=adapter._model_name)
    # gateway.run fixes its home at first import (another module may import it first), so the
    # real loader gets this test's config path explicitly.
    real_load = gateway_run._load_gateway_config
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
            patch("gateway.run._load_gateway_config", lambda: real_load(Path(_CONFIG))), \
            patch("run_agent.AIAgent") as agent_cls:
        adapter._create_agent(session_id="s1", **overrides)
    return agent_cls.call_args.kwargs["enabled_toolsets"]


def _configured(profile):
    return sorted(_get_platform_tools({"platform_toolsets": {"api_server": list(profile)}}, "api_server"))


class RequestCannotWidenTest(unittest.TestCase):
    def test_composite_plus_terminal_gives_only_the_configured_set(self):
        profile = ["hermes-api-server"]
        got = _effective_toolsets(profile, {"enabled_toolsets": ["hermes-api-server", "terminal"]})
        self.assertEqual(got, _configured(profile))

    def test_tool_outside_the_profile_is_not_added(self):
        got = _effective_toolsets(["web", "file"], {"enabled_toolsets": ["web", "terminal"]})
        self.assertEqual(got, ["web"])


class NoNarrowingKeepsConfiguredSetTest(unittest.TestCase):
    def test_empty_list_and_absent_field_keep_the_configured_set(self):
        profile = ["hermes-api-server"]
        self.assertEqual(_effective_toolsets(profile, {"enabled_toolsets": []}), _configured(profile))
        self.assertEqual(_effective_toolsets(profile, {}), _configured(profile))


class SubsetRequestTest(unittest.TestCase):
    def test_subset_request_gives_the_subset(self):
        got = _effective_toolsets(["hermes-api-server"], {"enabled_toolsets": ["web", "file"]})
        self.assertEqual(got, ["file", "web"])

    def test_dropping_memory_removes_it(self):
        profile = ["hermes-api-server"]
        keep = [t for t in _configured(profile) if t != "memory"]
        got = _effective_toolsets(profile, {"enabled_toolsets": keep})
        self.assertNotIn("memory", got)
        self.assertEqual(got, keep)


class ResolvedToolNamesStayInsideTest(unittest.TestCase):
    def test_resolved_tool_names_are_a_subset_of_the_configured_tools(self):
        profiles = (["hermes-api-server"], ["web", "file"], ["file"])
        requests = (["hermes-cli"], ["hermes-api-server", "terminal"], ["terminal", "browser", "web"],
                    ["web"], ["bogus"])
        for profile in profiles:
            allowed = _select_tool_names(_configured(profile), None, quiet_mode=True)
            for req in requests:
                with self.subTest(profile=profile, request=req):
                    got = _effective_toolsets(profile, {"enabled_toolsets": req})
                    self.assertLessEqual(_select_tool_names(got, None, quiet_mode=True), allowed)
                    self.assertLessEqual(set(got), set(_configured(profile)))


if __name__ == "__main__":
    unittest.main()
