import unittest
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.codegen_diff_utils import parse_git_numstat_output


class ParseGitNumstatOutputTests(unittest.TestCase):
    def test_parses_insertions_and_deletions(self) -> None:
        parsed = parse_git_numstat_output("5\t2\tsrc/file.py\n1\t0\tREADME.md\n")
        self.assertEqual(parsed["src/file.py"]["insertions"], 5)
        self.assertEqual(parsed["src/file.py"]["deletions"], 2)
        self.assertEqual(parsed["README.md"]["insertions"], 1)
        self.assertEqual(parsed["README.md"]["deletions"], 0)

    def test_handles_binary_or_malformed_counts(self) -> None:
        parsed = parse_git_numstat_output("-\t-\tassets/logo.png\nbad-line\n")
        self.assertEqual(parsed["assets/logo.png"]["insertions"], 0)
        self.assertEqual(parsed["assets/logo.png"]["deletions"], 0)
        self.assertNotIn("bad-line", parsed)


if __name__ == "__main__":
    unittest.main()
