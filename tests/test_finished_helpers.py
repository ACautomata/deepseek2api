import ast
import pathlib
import unittest
from typing import cast


APP_PATH = pathlib.Path(__file__).resolve().parents[1] / "app.py"


def load_app_functions(*names: str):
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))
    selected_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    selected_names = {node.name for node in selected_functions}
    missing = [name for name in names if name not in selected_names]
    if missing:
        raise AssertionError(f"Missing functions in app.py: {missing}")

    module = ast.Module(
        body=[cast(ast.stmt, node) for node in selected_functions], type_ignores=[]
    )
    namespace = {}
    exec("from typing import Any\n", namespace)
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


class DeepSeekFinishedHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            is_deepseek_finished_value,
            strip_deepseek_finished_suffix,
            split_stream_text_for_finished_suffix,
        ) = load_app_functions(
            "is_deepseek_finished_value",
            "strip_deepseek_finished_suffix",
            "split_stream_text_for_finished_suffix",
        )
        cls.is_deepseek_finished_value = staticmethod(is_deepseek_finished_value)
        cls.strip_deepseek_finished_suffix = staticmethod(
            strip_deepseek_finished_suffix
        )
        cls.split_stream_text_for_finished_suffix = staticmethod(
            split_stream_text_for_finished_suffix
        )

    def test_detects_string_finished_marker(self):
        self.assertTrue(self.is_deepseek_finished_value("FINISHED"))
        self.assertTrue(self.is_deepseek_finished_value("  FINISHED  "))

    def test_detects_status_list_finished_marker(self):
        self.assertTrue(
            self.is_deepseek_finished_value(
                [{"p": "status", "v": "FINISHED"}, {"p": "other", "v": "x"}]
            )
        )

    def test_ignores_normal_content(self):
        self.assertFalse(self.is_deepseek_finished_value("Hello"))
        self.assertFalse(
            self.is_deepseek_finished_value([{"p": "status", "v": "PENDING"}])
        )
        self.assertFalse(
            self.is_deepseek_finished_value({"p": "status", "v": "FINISHED"})
        )

    def test_strips_only_terminal_finished_suffix(self):
        self.assertEqual(
            self.strip_deepseek_finished_suffix('{"tool_calls": []}FINISHED'),
            '{"tool_calls": []}',
        )
        self.assertEqual(
            self.strip_deepseek_finished_suffix('{"tool_calls": []} FINISHED  '),
            '{"tool_calls": []}',
        )
        self.assertEqual(
            self.strip_deepseek_finished_suffix("FINISHED but not terminal content"),
            "FINISHED but not terminal content",
        )

    def test_stream_split_holds_possible_finished_suffix(self):
        emitted, pending = self.split_stream_text_for_finished_suffix("", "helloFIN")
        self.assertEqual(emitted, "hello")
        self.assertEqual(pending, "FIN")

        emitted, pending = self.split_stream_text_for_finished_suffix(pending, "ISHED")
        self.assertEqual(emitted, "")
        self.assertEqual(pending, "FINISHED")

    def test_stream_split_flushes_non_suffix_content(self):
        emitted, pending = self.split_stream_text_for_finished_suffix("FIN", "ISHX")
        self.assertEqual(emitted, "FINISHX")
        self.assertEqual(pending, "")


if __name__ == "__main__":
    unittest.main()
