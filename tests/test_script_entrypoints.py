"""Regression: mọi script CLI trong scripts/ phải chạy được bằng đường dẫn file.

Khi các script được dời từ thư mục gốc vào scripts/<nhóm>/, sys.path[0] trở
thành thư mục của script nên `import config`, `from modules...` và
`from scripts...` hỏng nếu thiếu shim thêm project root. smoke_carla_stack.py
còn tự spawn worker bằng đường dẫn file, nên thiếu shim là worker chết ngay.

Chỉ kiểm tra `--help`: argparse thoát trước khi chạm CARLA hay GPU. Script
không dùng argparse bị bỏ qua (KHÔNG chạy), vì với chúng `--help` bị phớt lờ và
chạy là thực thi thật (kết nối CARLA, xuất TensorRT, ghi file vào output/).
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROJECT_IMPORT_ERRORS = ("No module named 'modules'", "No module named 'scripts'",
                         "No module named 'config'")


def cli_scripts():
    return sorted(path for path in (ROOT / "scripts").rglob("*.py")
                  if path.name != "__init__.py"
                  and "argparse" in path.read_text(encoding="utf-8"))


class ScriptEntrypointTests(unittest.TestCase):
    def test_every_cli_script_resolves_project_imports_when_run_by_path(self):
        scripts = cli_scripts()
        self.assertGreater(len(scripts), 20)
        # cwd ngoài project: script không được dựa vào thư mục hiện hành.
        with tempfile.TemporaryDirectory() as cwd:
            for script in scripts:
                with self.subTest(script=str(script.relative_to(ROOT))):
                    result = subprocess.run(
                        [sys.executable, str(script), "--help"], cwd=cwd,
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=180)
                    # Môi trường CI không có carla/torch: thiếu gói bên thứ ba
                    # thì chấp nhận, nhưng gói của chính project phải import được.
                    for message in PROJECT_IMPORT_ERRORS:
                        self.assertNotIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
