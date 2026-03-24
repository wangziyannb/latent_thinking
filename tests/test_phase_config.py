import unittest

from methods.phase_config import PhaseOptions, resolve_phase_config


class PhaseConfigTest(unittest.TestCase):
    def test_legacy_disable_thinking_maps_inherit_to_no_think(self):
        cfg = resolve_phase_config(
            phase_name="latent",
            options=PhaseOptions(),
            task="math500",
            question="What is 2+2?",
            answer_only=False,
            legacy_disable_thinking=True,
        )
        self.assertEqual(cfg.thinking_mode, "no_think")
        self.assertFalse(cfg.enable_thinking)

    def test_prompt_preset_renders_placeholders(self):
        cfg = resolve_phase_config(
            phase_name="decode",
            options=PhaseOptions(
                prompt_preset="actor",
                system_prompt_override="Actor for {task_label}",
                user_prompt_override="Solve {question} on {task}.",
            ),
            task="math500",
            question="Find x.",
            answer_only=False,
        )
        self.assertEqual(cfg.system_prompt, "Actor for MATH-500")
        self.assertEqual(cfg.user_prompt, "Solve Find x. on math500.")

    def test_strip_think_tags_requires_no_think(self):
        with self.assertRaises(ValueError):
            resolve_phase_config(
                phase_name="decode",
                options=PhaseOptions(thinking_mode="think", strip_think_tags=True),
                task="gsm8k",
                question="What is 3+4?",
                answer_only=False,
            )


if __name__ == "__main__":
    unittest.main()
