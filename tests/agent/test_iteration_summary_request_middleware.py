"""The max-iterations summary request runs through the same ``llm_request`` middleware as a normal
turn call. Without it a session model that only a middleware resolves (for example the router alias
``auto`` of an auto-route plugin) goes to the provider unchanged, and the provider answers 400
("could not auto resolve a provider"); the user then gets ``max_iterations_no_summary``."""

import pytest

from agent import chat_completion_helpers as cch
from hermes_cli import plugins
from hermes_cli.plugins import PluginManager
from run_agent import AIAgent

_PLUGIN = """
SEEN = []

def register(ctx):
    def route(request=None, model='', **kw):
        SEEN.append({'model': model, 'request_model': (request or {}).get('model'),
                     'call_role': kw.get('call_role'), 'api_call_count': kw.get('api_call_count')})
        if (request or {}).get('model') != 'auto':
            return None
        return {'request': {**request, 'model': 'tier-medium'}, 'source': 'summary-route'}
    ctx.register_middleware('llm_request', route)
"""

_HISTORY = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "t1", "content": "r"},
]


@pytest.fixture
def seen(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "summary_route"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: summary_route\nversion: 0.1.0\n")
    (plugin / "__init__.py").write_text(_PLUGIN)
    (home / "config.yaml").write_text("plugins:\n  enabled: [summary_route]\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    return manager._plugins["summary_route"].module.SEEN


@pytest.fixture
def sent(monkeypatch):
    """Capture the request that the summary call sends; answer with a fixed summary."""
    requests = []

    def _fake_call(agent, api_request_id, request, callback, *, retry_count):
        requests.append(dict(request))
        return object()

    monkeypatch.setattr(cch, "_managed_summary_call", _fake_call)
    monkeypatch.setattr(cch, "_summary_text", lambda agent, response, **kw: "summary text")
    return requests


def _agent(model="auto"):
    agent = AIAgent(api_key="k", base_url="https://bifrost.example/v1", provider="custom", model=model,
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = "SYS"
    return agent


def test_summary_request_model_is_replaced_by_the_llm_request_middleware(seen, sent):
    agent = _agent("auto")
    assert agent.api_mode == "chat_completions"
    out = cch.handle_max_iterations(agent, [dict(m) for m in _HISTORY], 7)
    assert out == "summary text"
    assert len(sent) == 1
    assert sent[0]["model"] == "tier-medium"           # without the fix: "auto"
    assert seen and seen[-1]["call_role"] == "iteration_summary"
    assert seen[-1]["api_call_count"] == 7
    assert seen[-1]["model"] == "auto"


def test_summary_request_is_unchanged_when_the_middleware_returns_none(seen, sent):
    agent = _agent("real-model")
    cch.handle_max_iterations(agent, [dict(m) for m in _HISTORY], 3)
    assert sent[0]["model"] == "real-model"
    assert seen  # the middleware ran


def test_a_failing_middleware_sends_the_summary_unmodified(sent, monkeypatch):
    import hermes_cli.middleware as mw

    def _boom(*a, **kw):
        raise RuntimeError("middleware broke")

    monkeypatch.setattr(mw, "apply_llm_request_middleware", _boom)
    agent = _agent("auto")
    out = cch.handle_max_iterations(agent, [dict(m) for m in _HISTORY], 2)
    assert out == "summary text"
    assert sent[0]["model"] == "auto"
