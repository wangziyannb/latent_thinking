import unittest

import torch

from methods.latent_self_think import LatentSelfThink, RunConfig


class FakeModel:
    def __init__(self):
        self.prepare_calls = []
        self.append_calls = []

    def prepare_chat_input(self, messages, add_generation_prompt=True, *, enable_thinking=None, strip_think_tags=False):
        self.prepare_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "strip_think_tags": strip_think_tags,
            }
        )
        return "prompt", torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]])

    def generate_latent_batch(self, **kwargs):
        return "latent_cache", False, 1, []

    def append_text_to_past(self, text, past_key_values):
        self.append_calls.append((text, past_key_values))
        return f"{past_key_values}|{text}"

    def generate_text_batch(self, **kwargs):
        return [r"The final answer is \boxed{42}."], None, torch.tensor([5])


class LatentSelfThinkTest(unittest.TestCase):
    def test_phase_specific_prompts_and_modes(self):
        model = FakeModel()
        runner = LatentSelfThink(
            model,
            RunConfig(
                task="math500",
                latent_steps=1,
                latent_thinking_mode="think",
                decode_thinking_mode="no_think",
                latent_prompt_preset="thinker",
                decode_prompt_preset="actor",
                close_latent_think_tag_before_decode=True,
                strategy_label="Think + No Think",
            ),
        )

        out = runner.run_one(
            {
                "task": "math500",
                "question": "Compute 40 + 2.",
                "gold": "42",
            }
        )

        self.assertEqual(model.prepare_calls[0]["enable_thinking"], True)
        self.assertEqual(model.prepare_calls[1]["enable_thinking"], False)
        self.assertIn("Thinker agent", model.prepare_calls[0]["messages"][0]["content"])
        self.assertIn("Actor agent", model.prepare_calls[1]["messages"][0]["content"])
        self.assertEqual(model.append_calls, [("</think>", "latent_cache")])
        self.assertTrue(out["closed_latent_think_tag_before_decode"])
        self.assertEqual(out["strategy_label"], "Think + No Think")
        self.assertEqual(out["phase_configs"]["latent"]["thinking_mode"], "think")
        self.assertEqual(out["phase_configs"]["decode"]["thinking_mode"], "no_think")


if __name__ == "__main__":
    unittest.main()
