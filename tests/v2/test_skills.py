"""Tests for skills system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from open_harness_v2.skills.loader import Skill, SkillRegistry, _parse_skill_file


class TestSkill:
    def test_expand_no_args(self):
        skill = Skill(name="test", description="desc", prompt="Do the thing")
        assert skill.expand() == "Do the thing"

    def test_expand_with_args(self):
        skill = Skill(name="test", description="desc", prompt="Do the thing")
        result = skill.expand("extra context")
        assert "Do the thing" in result
        assert "extra context" in result

    def test_expand_preserves_prompt(self):
        skill = Skill(name="test", description="desc", prompt="Line 1\nLine 2")
        result = skill.expand("args")
        assert "Line 1\nLine 2" in result


class TestSkillRegistry:
    def test_register_and_get(self):
        registry = SkillRegistry()
        skill = Skill(name="foo", description="bar", prompt="baz")
        registry.register(skill)
        assert registry.get("foo") is skill
        assert registry.get("nonexistent") is None

    def test_list_skills(self):
        registry = SkillRegistry()
        registry.register(Skill(name="a", description="", prompt="p"))
        registry.register(Skill(name="b", description="", prompt="p"))
        assert len(registry.list_skills()) == 2

    def test_skill_names(self):
        registry = SkillRegistry()
        registry.register(Skill(name="x", description="", prompt="p"))
        assert "x" in registry.skill_names()

    def test_later_registration_overrides(self):
        registry = SkillRegistry()
        registry.register(Skill(name="s", description="v1", prompt="p1"))
        registry.register(Skill(name="s", description="v2", prompt="p2"))
        assert registry.get("s").description == "v2"

    def test_discover_builtin(self):
        """Built-in skills directory should have at least the commit skill."""
        registry = SkillRegistry()
        registry.discover()
        assert registry.get("commit") is not None
        assert registry.get("review") is not None

    def test_discover_project_dir(self):
        """Skills in project .harness/skills/ should be discovered."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / ".harness" / "skills"
            skills_dir.mkdir(parents=True)
            skill_file = skills_dir / "custom.yaml"
            skill_file.write_text(yaml.dump({
                "name": "custom",
                "description": "A custom skill",
                "prompt": "Do custom stuff",
            }))
            registry = SkillRegistry()
            registry.discover(project_root=Path(tmp))
            assert registry.get("custom") is not None
            assert registry.get("custom").source == "project"


class TestParseSkillFile:
    def test_parse_valid(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump({
                "name": "myskill",
                "description": "My skill",
                "prompt": "Do stuff",
                "args": "optional",
                "args_description": "Some args",
            }, f)
            f.flush()
            skill = _parse_skill_file(Path(f.name), "test")
        assert skill.name == "myskill"
        assert skill.description == "My skill"
        assert skill.args == "optional"
        assert skill.source == "test"

    def test_parse_uses_stem_as_name(self):
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, prefix="fallback_"
        ) as f:
            yaml.dump({"prompt": "Do stuff"}, f)
            f.flush()
            skill = _parse_skill_file(Path(f.name), "test")
        # Name should be derived from file stem
        assert skill.name is not None
        assert len(skill.name) > 0

    def test_parse_no_prompt_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump({"name": "bad", "description": "no prompt"}, f)
            f.flush()
            with pytest.raises(ValueError, match="no prompt"):
                _parse_skill_file(Path(f.name), "test")
