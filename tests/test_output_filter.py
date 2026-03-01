"""Tests for Issue 6: Sensitive data masking in tool output."""

from open_harness.tools.output_filter import redact_secrets


class TestOutputFilter:
    def test_empty_string(self):
        assert redact_secrets("") == ""

    def test_no_secrets(self):
        text = "Hello world, this is normal output."
        assert redact_secrets(text) == text

    def test_redact_api_key(self):
        text = "api_key=sk-abc123def456ghi789jklmnop"
        result = redact_secrets(text)
        assert "sk-abc123def456ghi789jklmnop" not in result
        assert "REDACTED" in result or "API_KEY" in result

    def test_redact_openai_key_pattern(self):
        text = "Found key: sk-abcdef1234567890abcdef"
        result = redact_secrets(text)
        assert "sk-abcdef1234567890abcdef" not in result
        assert "API_KEY" in result

    def test_redact_github_pat(self):
        text = "Found ghp_abcdef1234567890abcdef1234567890abcdef in output"
        result = redact_secrets(text)
        assert "ghp_abcdef1234567890abcdef1234567890abcdef" not in result
        assert "GITHUB_PAT" in result

    def test_redact_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"
        result = redact_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "REDACTED" in result

    def test_redact_password_in_env(self):
        text = "PASSWORD=super_secret_pass123"
        result = redact_secrets(text)
        assert "super_secret_pass123" not in result
        assert "REDACTED" in result

    def test_redact_aws_credentials(self):
        text = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = redact_secrets(text)
        assert "wJalrXUtnFEMI" not in result
        assert "REDACTED" in result

    def test_redact_token_key_value(self):
        text = "token: abc123def456"
        result = redact_secrets(text)
        assert "abc123def456" not in result

    def test_preserves_normal_text_around_secrets(self):
        text = "Before SECRET=mysecret After"
        result = redact_secrets(text)
        assert "Before" in result
        assert "After" in result
        assert "mysecret" not in result

    def test_multiple_secrets(self):
        text = "api_key=key1 password=pass2 token=tok3"
        result = redact_secrets(text)
        assert "key1" not in result
        assert "pass2" not in result
        assert "tok3" not in result

    def test_database_url(self):
        text = "DATABASE_URL=postgres://user:pass@host:5432/db"
        result = redact_secrets(text)
        assert "postgres://user:pass" not in result
        assert "REDACTED" in result
