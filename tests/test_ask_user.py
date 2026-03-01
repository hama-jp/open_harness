"""Tests for Issue 5: AskUserTool."""

from open_harness.tools.file_ops import AskUserTool


class TestAskUserTool:
    def test_no_callback_returns_error(self):
        tool = AskUserTool(user_input_fn=None)
        result = tool.execute(question="What color?")
        assert not result.success
        assert "non-interactive" in result.error

    def test_with_callback(self):
        tool = AskUserTool(user_input_fn=lambda q: "blue")
        result = tool.execute(question="What color?")
        assert result.success
        assert result.output == "blue"

    def test_callback_receives_question(self):
        received = []
        def capture(q):
            received.append(q)
            return "answer"

        tool = AskUserTool(user_input_fn=capture)
        tool.execute(question="What is your name?")
        assert received == ["What is your name?"]

    def test_empty_question(self):
        tool = AskUserTool(user_input_fn=lambda q: "x")
        result = tool.execute(question="")
        assert not result.success
        assert "No question" in result.error

    def test_callback_exception(self):
        def fail(q):
            raise RuntimeError("broken")

        tool = AskUserTool(user_input_fn=fail)
        result = tool.execute(question="test")
        assert not result.success
        assert "broken" in result.error

    def test_keyboard_interrupt(self):
        def interrupt(q):
            raise KeyboardInterrupt()

        tool = AskUserTool(user_input_fn=interrupt)
        result = tool.execute(question="test")
        assert result.success
        assert "declined" in result.output

    def test_callback_returns_none(self):
        """None return from callback should be handled gracefully."""
        tool = AskUserTool(user_input_fn=lambda q: None)
        result = tool.execute(question="test")
        assert result.success
        assert result.output == ""

    def test_callback_returns_int(self):
        """Integer return from callback should be converted to string."""
        tool = AskUserTool(user_input_fn=lambda q: 42)
        result = tool.execute(question="test")
        assert result.success
        assert result.output == "42"

    def test_schema(self):
        tool = AskUserTool()
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "ask_user"
        assert "question" in schema["function"]["parameters"]["properties"]
