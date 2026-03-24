import unittest

from models import _strip_assistant_tail_think_tags


class ModelPromptSanitizerTest(unittest.TestCase):
    def test_strip_open_tag_at_tail(self):
        self.assertEqual(
            _strip_assistant_tail_think_tags("assistant\n<think>"),
            "assistant",
        )

    def test_strip_closed_pair_at_tail(self):
        self.assertEqual(
            _strip_assistant_tail_think_tags("assistant\n<think></think>\n"),
            "assistant\n",
        )

    def test_keep_non_tail_tags(self):
        text = "user mentioned <think>literal</think> in the prompt.\nassistant"
        self.assertEqual(_strip_assistant_tail_think_tags(text), text)


if __name__ == "__main__":
    unittest.main()
