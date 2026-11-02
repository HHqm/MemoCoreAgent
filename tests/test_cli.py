import contextlib
import io
import unittest

from memocore.cli import build_parser, main


class CliTest(unittest.TestCase):
    def test_run_defaults(self):
        args = build_parser().parse_args(["run", "--system", "naive_rag"])
        self.assertEqual(args.categories, (1, 2, 3, 4))
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.judge, "strict")
        self.assertEqual(build_parser().parse_args(["run", "--system", "naive_rag", "--categories", "1,5"]).categories, (1, 5))

    def test_user_errors_print_a_message_instead_of_a_traceback(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["report", "--run-name", "no-such-run-for-tests"])
        self.assertEqual(code, 1)
        self.assertIn("no-such-run-for-tests", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
