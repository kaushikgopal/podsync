"""Tests for CLI interface."""

from click.testing import CliRunner

from podsync.cli import main


class TestCliArguments:
    def test_requires_master_flag(self):
        """CLI requires --master flag."""
        runner = CliRunner()
        # Invoke with no arguments to check required flag handling
        result = runner.invoke(main, [])

        assert result.exit_code != 0
        assert 'master' in result.output.lower() or 'missing' in result.output.lower()

    def test_requires_tracks_flag(self):
        """CLI requires --tracks flag."""
        runner = CliRunner()
        # Create a temp file for master to isolate the tracks requirement
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'\x00' * 100)  # dummy content
            temp_path = f.name

        try:
            result = runner.invoke(main, ['--master', temp_path])
            assert result.exit_code != 0
            assert 'tracks' in result.output.lower() or 'missing' in result.output.lower()
        finally:
            import os
            os.unlink(temp_path)

    def test_accepts_all_options(self):
        """CLI accepts all documented options."""
        runner = CliRunner()
        # Use --help to verify options are recognized
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert '--master' in result.output
        assert '--tracks' in result.output
        assert '--sync-window' in result.output
        assert '--output-suffix' in result.output
