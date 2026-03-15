"""Skills system — reusable slash commands inspired by Claude Code.

Skills are prompt templates that expand into full agent goals when
invoked via ``/skill-name`` in the REPL.  They can be:

  - Built-in (packaged with Open Harness)
  - Project-level (``.harness/skills/``)
  - User-level (``~/.config/open-harness/skills/``)

Each skill is a YAML file with the following structure::

    name: commit
    description: "Stage and commit changes with a descriptive message"
    prompt: |
      Look at the current git diff and status.
      Create a well-structured commit with a descriptive message.
      Follow conventional commit format.
    args: optional  # "none", "optional", "required"
    args_description: "Optional commit message override"
"""

from open_harness_v2.skills.loader import Skill, SkillRegistry

__all__ = ["Skill", "SkillRegistry"]
