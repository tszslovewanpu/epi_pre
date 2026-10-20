"""Exercise the upload check against real, isolated Git repositories."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_repository.py"


class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="epitope-hygiene-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copy2(ROOT / ".gitignore", self.root / ".gitignore")

    def write(self, name, content="small file\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args],
                              capture_output=True, check=True)

    def check(self, *args):
        return subprocess.run([sys.executable, str(CHECK), "--root", str(self.root), *args],
                              capture_output=True, text=True)

    def test_generated_artifacts_ignored_but_source_and_templates_visible(self):
        self.git("init", "--quiet")
        ignored = [
            "dataset/train.csv", "data/index.json", ".local/runs/metrics.json",
            "machine.local.json", ".env", ".env.production", "configs/config.local.yaml",
            "baselines/new/outputs_big/run/final_metrics.json", "model.pth",
            "baselines/_shared/new/unknown_artifact.txt", "predictions/summary.json",
            "inference/summary.json", "predictions_new/summary.json",
            "custom_experiment/config.resolved.json", "custom_experiment/run_metadata.json",
            "custom_experiment/eval_metadata_test.json", "custom_experiment/final_metrics.json",
        ]
        source = [
            ".env.example", "machine.example.json", "configs/large_gpu.yaml",
            "baselines/se3bench/config.yaml", "baselines/deepinteract/LICENSE",
        ]
        for name in ignored + source:
            self.write(name)
        candidates = set(self.git("ls-files", "--others", "--exclude-standard", "-z").stdout.decode().split("\0"))
        self.assertTrue(set(source) <= candidates)
        self.assertFalse(set(ignored) & candidates)
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_forced_tracked_artifact_is_rejected_without_printing_content(self):
        self.git("init", "--quiet")
        self.write(".env.production", "TOKEN=not-for-tool-output\n")
        self.write("checkpoints/best.pt")
        self.git("add", "-f", "--", ".env.production", "checkpoints/best.pt")
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(".env.production", result.stdout)
        self.assertIn("checkpoints/best.pt", result.stdout)
        self.assertNotIn("not-for-tool-output", result.stdout + result.stderr)

    def test_runtime_artifacts_rejected_even_without_ignore_rules(self):
        self.git("init", "--quiet")
        (self.root / ".gitignore").unlink()
        names = [
            "baselines/_shared/unknown.txt", "predictions/summary.json",
            "inference/summary.json", "inference_validation/summary.json",
            "custom_experiment/config.resolved.json", "custom_experiment/run_metadata.json",
            "custom_experiment/eval_metadata_test.json", "custom_experiment/final_metrics.json",
        ]
        for name in names:
            self.write(name)
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        for name in names:
            self.assertIn(name, result.stdout)

    def test_staged_large_blob_detected_after_working_file_shrinks(self):
        self.git("init", "--quiet")
        self.write("large.txt", "x" * 2000)
        self.git("add", "--", "large.txt")
        self.write("large.txt", "small")
        result = self.check("--max-size-mb", "0.001")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('FAIL "large.txt": staged/tracked blob exceeds size limit', result.stdout)

    def test_before_git_init_respects_ignore_and_leaves_no_git_directory(self):
        self.write("model.py")
        self.write(".local/huge.txt", "x" * 100000)
        self.write(".env", "TOKEN=private\n")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Git not initialized", result.stdout)
        self.assertFalse((self.root / ".git").exists())

    def test_large_untracked_file_and_newline_filename_are_safe(self):
        self.git("init", "--quiet")
        self.write("large\nfile.txt", "x" * 2000)
        result = self.check("--max-size-mb", "0.001")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('"large\\nfile.txt"', result.stdout)
        self.assertNotIn('"large\nfile.txt"', result.stdout)


if __name__ == "__main__":
    unittest.main()
