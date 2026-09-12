"""Static regression checks for the safe CARLA launcher modes.

The batch file is intentionally not executed by unit tests: launching CARLA is
an integration action and must be performed only with a user-owned server.
"""

from pathlib import Path
import unittest


LAUNCHER = Path(__file__).with_name("launch_carla.bat")


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = LAUNCHER.read_text(encoding="utf-8").lower()

    def test_documents_both_rhi_modes_and_legacy_off_mode(self):
        self.assertIn("launch_carla.bat default-rhi", self.text)
        self.assertIn("launch_carla.bat dx11", self.text)
        self.assertIn('if /i "%~1"=="off"', self.text)

    def test_default_rhi_clears_forced_dx11_argument(self):
        self.assertIn('set "rhi_mode=default-rhi"', self.text)
        self.assertIn('set "rhi_args="', self.text)
        self.assertIn('if /i "%~1"=="dx11" (', self.text)
        self.assertIn('set "rhi_args=-d3d11"', self.text)
        self.assertIn('carlaue4.exe" %rhi_args%', self.text)

    def test_no_argument_defaults_to_default_rhi(self):
        self.assertIn(
            'set "rhi_mode=default-rhi"\nset "rhi_args="',
            self.text,
        )
        self.assertIn("launch_carla.bat          -> rhi mac dinh", self.text)

    def test_unknown_arguments_fail_before_launch(self):
        self.assertIn('set "arg_error=%~1"', self.text)
        self.assertIn("exit /b 5", self.text)
        self.assertIn("if defined arg_error", self.text)


if __name__ == "__main__":
    unittest.main()
