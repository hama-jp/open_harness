"""Tests for Issue 11: ProjectTreeTool."""

import os
import shutil
import tempfile

from open_harness.tools.file_ops import ProjectTreeTool


class TestProjectTreeTool:
    def setup_method(self):
        self.tool = ProjectTreeTool()
        self.tmpdir = tempfile.mkdtemp()
        # Create structure:
        # tmpdir/
        #   src/
        #     main.py
        #     util.py
        #   docs/
        #     readme.txt
        #   .git/        (should be skipped)
        #     config
        #   __pycache__/ (should be skipped)
        #     foo.pyc
        os.makedirs(os.path.join(self.tmpdir, "src"))
        os.makedirs(os.path.join(self.tmpdir, "docs"))
        os.makedirs(os.path.join(self.tmpdir, ".git"))
        os.makedirs(os.path.join(self.tmpdir, "__pycache__"))
        open(os.path.join(self.tmpdir, "src", "main.py"), "w").close()
        open(os.path.join(self.tmpdir, "src", "util.py"), "w").close()
        open(os.path.join(self.tmpdir, "docs", "readme.txt"), "w").close()
        open(os.path.join(self.tmpdir, ".git", "config"), "w").close()
        open(os.path.join(self.tmpdir, "__pycache__", "foo.pyc"), "w").close()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_basic_tree(self):
        result = self.tool.execute(path=self.tmpdir, max_depth=3)
        assert result.success
        assert "src/" in result.output
        assert "docs/" in result.output
        assert "main.py" in result.output

    def test_skips_git_and_pycache(self):
        result = self.tool.execute(path=self.tmpdir, max_depth=3)
        assert ".git" not in result.output
        assert "__pycache__" not in result.output
        assert "foo.pyc" not in result.output

    def test_max_depth_1(self):
        result = self.tool.execute(path=self.tmpdir, max_depth=1)
        assert result.success
        # Should show dirs but not their contents
        assert "src/" in result.output
        assert "main.py" not in result.output

    def test_nonexistent_path(self):
        result = self.tool.execute(path="/tmp/nonexistent_dir_12345")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_file_not_dir(self):
        path = os.path.join(self.tmpdir, "src", "main.py")
        result = self.tool.execute(path=path)
        assert not result.success
        assert "Not a directory" in result.error

    def test_tree_format(self):
        result = self.tool.execute(path=self.tmpdir, max_depth=2)
        # Should contain tree characters
        lines = result.output.split("\n")
        assert any("\u251c" in line or "\u2514" in line for line in lines)

    def test_schema(self):
        schema = self.tool.to_openai_schema()
        assert schema["function"]["name"] == "project_tree"
        params = schema["function"]["parameters"]["properties"]
        assert "path" in params
        assert "max_depth" in params
