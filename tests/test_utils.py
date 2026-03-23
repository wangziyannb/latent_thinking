import unittest

from utils import (
    answers_match,
    extract_boxed_text,
    extract_math_answer,
    normalize_math_answer,
)


class UtilsTest(unittest.TestCase):
    def test_extract_boxed_text_handles_nested_braces(self):
        text = r"Therefore, the final answer is: \boxed{\frac{14}{3}}."
        self.assertEqual(extract_boxed_text(text), r"\frac{14}{3}")

    def test_extract_math_answer_prefers_boxed_content(self):
        text = r"Work... Therefore, the final answer is: \boxed{\left( 3, \frac{\pi}{2} \right)}."
        self.assertEqual(extract_math_answer(text), r"\left( 3, \frac{\pi}{2} \right)")

    def test_normalize_math_answer_removes_formatting_noise(self):
        self.assertEqual(
            normalize_math_answer(r"\left( 3, \frac{\pi}{2} \right)"),
            r"(3,\frac{\pi}{2})",
        )

    def test_answers_match_for_math500_equivalent_formatting(self):
        self.assertTrue(
            answers_match(
                "math500",
                r"(3,\frac{\pi}{2})",
                r"\left( 3, \frac{\pi}{2} \right)",
            )
        )

    def test_answers_match_for_math500_text_answer(self):
        self.assertTrue(answers_match("math500", "Evelyn", r"\text{Evelyn}"))

    def test_answers_match_for_gsm8k_numeric(self):
        self.assertTrue(answers_match("gsm8k", "42", "42"))


if __name__ == "__main__":
    unittest.main()
