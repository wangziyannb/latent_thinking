import unittest

import torch

from methods.latent_self_think import LatentSelfThink, RunConfig


class FakeModel:
    def __init__(self):
        self.prepare_calls = []
        self.append_calls = []
        self.text_batch_sizes = []
        self.latent_batch_sizes = []

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

    def prepare_chat_batch(self, batch_messages, add_generation_prompt=True, *, enable_thinking=None, strip_think_tags=False):
        self.prepare_calls.append(
            {
                "messages": batch_messages,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "strip_think_tags": strip_think_tags,
                "batch_size": len(batch_messages),
            }
        )
        input_ids = torch.tensor([[1, 2, 3] for _ in batch_messages])
        attention_mask = torch.ones_like(input_ids)
        return ["prompt" for _ in batch_messages], input_ids, attention_mask

    def generate_latent_batch(self, **kwargs):
        self.latent_batch_sizes.append(int(kwargs["input_ids"].shape[0]))
        return "latent_cache", False, 1, []

    def append_text_to_past(self, text, past_key_values):
        self.append_calls.append((text, past_key_values))
        return f"{past_key_values}|{text}"

    def generate_text_batch(self, **kwargs):
        batch_size = int(kwargs["input_ids"].shape[0])
        self.text_batch_sizes.append(batch_size)
        outputs = []
        for idx in range(batch_size):
            outputs.append(rf"The final answer is \boxed{{{42 + idx}}}.")
        return outputs, None, torch.tensor([5 for _ in range(batch_size)])


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
        self.assertIn("Thinker agent", model.prepare_calls[0]["messages"][0][0]["content"])
        self.assertIn("Actor agent", model.prepare_calls[1]["messages"][0][0]["content"])
        self.assertEqual(model.append_calls, [("</think>", "latent_cache")])
        self.assertTrue(out["closed_latent_think_tag_before_decode"])
        self.assertEqual(out["strategy_label"], "Think + No Think")
        self.assertEqual(out["phase_configs"]["latent"]["thinking_mode"], "think")
        self.assertEqual(out["phase_configs"]["decode"]["thinking_mode"], "no_think")
        self.assertEqual(model.text_batch_sizes, [1])

    def test_run_batch_non_early_stop_uses_batched_calls(self):
        model = FakeModel()
        runner = LatentSelfThink(
            model,
            RunConfig(
                task="math500",
                latent_steps=1,
            ),
        )

        outs = runner.run_batch(
            [
                {"task": "math500", "question": "Compute 40 + 2.", "gold": "42"},
                {"task": "math500", "question": "Compute 40 + 3.", "gold": "43"},
            ]
        )

        self.assertEqual(len(outs), 2)
        self.assertEqual(model.latent_batch_sizes, [2])
        self.assertEqual(model.text_batch_sizes, [2])
        self.assertEqual(outs[0]["prediction"], "42")
        self.assertEqual(outs[1]["prediction"], "43")

    def test_run_batch_falls_back_to_single_item_when_early_stop_enabled(self):
        model = FakeModel()
        runner = LatentSelfThink(
            model,
            RunConfig(
                task="math500",
                latent_steps=1,
                latent_early_stop=True,
            ),
        )

        outs = runner.run_batch(
            [
                {"task": "math500", "question": "Compute 40 + 2.", "gold": "42"},
                {"task": "math500", "question": "Compute 40 + 3.", "gold": "42"},
            ]
        )

        self.assertEqual(len(outs), 2)
        self.assertEqual(model.latent_batch_sizes, [1, 1])
        self.assertEqual(model.text_batch_sizes, [1, 1])


if __name__ == "__main__":
    unittest.main()
