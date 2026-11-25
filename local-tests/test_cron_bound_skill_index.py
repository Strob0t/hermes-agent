"""Tests of the cron bound-skill index port (fork finding #18, erhnysr fix/cron-scope-skill-bound-index).

The patch scopes the skill index in the system prompt of a cron agent to the skills that its job loads.
``_load_cron_skill_parts`` collects the canonical names of the loaded skills, ``_resolve_bound_skills``
maps them to ``None`` (the job lists no skills: full index), ``[]`` or the list, and the cron agent gets
them as ``AIAgent(bound_skills=...)``. ``build_skills_system_prompt`` hides all other skills; the value
is part of its cache key. The tests show:
 - a cron job with skills [a, b] gets a system prompt whose skill index lists only a and b;
 - a cron job without skills and a normal session both keep the full skill index.
The tests use a temp HERMES_HOME with 3 skills and build real agents (no network call).
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

SKILLS = ("alpha-skill", "beta-skill", "gamma-skill")
_HOME = None
_OLD_HOME = None


def setUpModule():
    global _HOME, _OLD_HOME
    _HOME = tempfile.mkdtemp(prefix="hermes-bound-skills-")
    _OLD_HOME = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = _HOME
    for name in SKILLS:
        skill_dir = Path(_HOME, "skills", "misc", name)
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: The {name} test skill.\n---\nBody of {name}.\n")


def tearDownModule():
    if _OLD_HOME is None:
        os.environ.pop("HERMES_HOME", None)
    else:
        os.environ["HERMES_HOME"] = _OLD_HOME
    shutil.rmtree(_HOME, ignore_errors=True)


def _listed(system_prompt):
    return [name for name in SKILLS if name in system_prompt]


def _cron_system_prompt(job):
    """Run the cron prompt and agent build path of ``run_job`` and return the agent's system prompt."""
    from agent.system_prompt import build_system_prompt
    from cron.scheduler import _construct_cron_agent, _CronAgentSetup
    from cron.scheduler_prompt import _build_job_prompt, _resolve_bound_skills
    from run_agent import AIAgent

    bound_out = []
    _build_job_prompt(job, bound_skills_out=bound_out)
    setup = _CronAgentSetup(model="test/model", runtime={
        "api_key": "test-key", "base_url": "http://127.0.0.1:9/v1", "provider": "custom"})
    agent = _construct_cron_agent(AIAgent, job, {}, setup, workdir=None, session_id="cron_test",
                                  session_db=None, bound_skills=_resolve_bound_skills(job, bound_out))
    return build_system_prompt(agent)


class CronBoundSkillIndexTest(unittest.TestCase):
    def test_cron_job_with_skills_lists_only_its_skills(self):
        job = {"id": "job-ab", "name": "ab", "prompt": "Do the work.", "skills": ["alpha-skill", "beta-skill"]}
        self.assertEqual(_listed(_cron_system_prompt(job)), ["alpha-skill", "beta-skill"])

    def test_cron_job_without_skills_keeps_full_index(self):
        job = {"id": "job-none", "name": "none", "prompt": "Do the work."}
        self.assertEqual(_listed(_cron_system_prompt(job)), list(SKILLS))


class NormalSessionSkillIndexTest(unittest.TestCase):
    def test_normal_session_lists_all_skills(self):
        from agent.system_prompt import build_system_prompt
        from run_agent import AIAgent

        agent = AIAgent(model="test/model", api_key="test-key", base_url="http://127.0.0.1:9/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
        self.assertIsNone(getattr(agent, "bound_skills", None))
        self.assertEqual(_listed(build_system_prompt(agent)), list(SKILLS))


if __name__ == "__main__":
    unittest.main()
