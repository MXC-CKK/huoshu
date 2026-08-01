"""Tests for learning_agent.llm — LLM client configuration and fallback."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from learning_agent.llm import (
    LLMClient,
    LLMConfig,
    _parse_score_response,
    get_config_summary,
    is_llm_available,
)

# ── 模拟 openai 模块 ──────────────────────────────────────────────────


def _mock_openai_module() -> MagicMock:
    """创建一个模拟的 openai 模块。"""
    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Mocked reply from LLM."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai


# ── LLMConfig 测试 ────────────────────────────────────────────────────


class TestLLMConfig:
    """LLMConfig 测试。"""

    def test_defaults(self) -> None:
        """默认 provider=deepseek。"""
        cfg = LLMConfig()
        assert cfg.provider == "deepseek"
        assert cfg.model == "deepseek-chat"
        assert "deepseek.com" in cfg.base_url

    def test_from_env_defaults(self) -> None:
        """无环境变量时使用 deepseek 默认值。"""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig.from_env()
            assert cfg.provider == "deepseek"
            assert cfg.model == "deepseek-chat"

    def test_from_env_deepseek(self) -> None:
        """LLM_PROVIDER=deepseek + LLM_API_KEY。"""
        env = {"LLM_PROVIDER": "deepseek", "LLM_API_KEY": "sk-test-123"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig.from_env()
            assert cfg.provider == "deepseek"
            assert cfg.api_key == "sk-test-123"

    def test_from_env_openai(self) -> None:
        """LLM_PROVIDER=openai。"""
        env = {"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-openai"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig.from_env()
            assert cfg.provider == "openai"
            assert cfg.model == "gpt-4o-mini"
            assert "openai.com" in cfg.base_url

    def test_from_env_ollama(self) -> None:
        """LLM_PROVIDER=ollama。"""
        env = {"LLM_PROVIDER": "ollama"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig.from_env()
            assert cfg.provider == "ollama"
            assert "localhost" in cfg.base_url
            assert cfg.model == "llama3.2"

    def test_from_env_custom_base_url(self) -> None:
        """LLM_BASE_URL 覆盖默认。"""
        env = {"LLM_BASE_URL": "https://custom.api/v1", "LLM_API_KEY": "sk-xxx"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig.from_env()
            assert cfg.base_url == "https://custom.api/v1"

    def test_from_env_custom_model(self) -> None:
        """LLM_MODEL 覆盖默认。"""
        env = {"LLM_MODEL": "deepseek-reasoner", "LLM_API_KEY": "sk-xxx"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig.from_env()
            assert cfg.model == "deepseek-reasoner"


# ── LLMClient 测试 ────────────────────────────────────────────────────


class TestLLMClient:
    """LLMClient 测试。"""

    def test_from_env(self) -> None:
        """从环境变量创建。"""
        env = {"LLM_API_KEY": "sk-test", "LLM_PROVIDER": "deepseek"}
        with patch.dict(os.environ, env, clear=True):
            client = LLMClient.from_env()
            assert client.config.api_key == "sk-test"

    def test_available_with_key(self) -> None:
        """有 API key 时 available=True。"""
        client = LLMClient(config=LLMConfig(api_key="sk-xx"))
        assert client.available

    def test_available_ollama(self) -> None:
        """Ollama 本地模式 always available。"""
        client = LLMClient(config=LLMConfig(provider="ollama"))
        assert client.available

    def test_not_available_without_key(self) -> None:
        """无 API key 时 available=False。"""
        client = LLMClient(config=LLMConfig(api_key=""))
        assert not client.available

    def test_chat_without_key_raises(self) -> None:
        """无可用 LLM 时 chat() 抛出 RuntimeError。"""
        client = LLMClient(config=LLMConfig(api_key=""))
        with pytest.raises(RuntimeError, match="LLM 不可用"):
            client.chat([{"role": "user", "content": "hi"}])


class TestLLMClientMocked:
    """LLMClient 测试（mock OpenAI API via sys.modules 注入）。"""

    @pytest.fixture(autouse=True)
    def _setup(self, request: pytest.FixtureRequest) -> None:
        """注入模拟 openai 模块。"""
        self.mock_openai = _mock_openai_module()
        orig_openai = sys.modules.get("openai")
        sys.modules["openai"] = self.mock_openai

        def _cleanup() -> None:
            if orig_openai is not None:
                sys.modules["openai"] = orig_openai
            else:
                sys.modules.pop("openai", None)

        request.addfinalizer(_cleanup)

    @pytest.fixture
    def client(self) -> LLMClient:
        return LLMClient(config=LLMConfig(
            provider="deepseek",
            api_key="sk-test",
            base_url="https://test.api/v1",
            model="test-model",
        ))

    def test_chat_returns_content(self, client: LLMClient) -> None:
        """chat() 返回模型回复文本。"""
        reply = client.chat([{"role": "user", "content": "hello"}])
        assert reply == "Mocked reply from LLM."

    def test_chat_sets_custom_content(self, client: LLMClient) -> None:
        """chat() 可以设置自定义回复。"""
        mock_choice = MagicMock()
        mock_choice.message.content = "Custom reply."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client

        reply = client.chat([{"role": "user", "content": "hello"}])
        assert reply == "Custom reply."

    def test_socratic_teach(self, client: LLMClient) -> None:
        """socratic_teach() 构建 system prompt 并调用。"""
        mock_choice = MagicMock()
        mock_choice.message.content = "Let's think about this step by step..."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client

        ctx = {
            "item": {"id": "a", "title": "Test", "type": "concept", "mode": "whitebox", "source": "§1", "mastery": 0.5},
            "prerequisites": [],
            "related": [],
            "instruction": "Test instruction",
        }
        reply = client.socratic_teach(ctx, "What is this?")
        assert len(reply) > 0
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"

    def test_generate_questions(self, client: LLMClient) -> None:
        """generate_questions() 返回题目列表。"""
        mock_choice = MagicMock()
        mock_choice.message.content = "1. Question one?\n2. Question two?\n3. Question three?"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client

        ctx = {
            "item": {"id": "a", "title": "Sample Mean", "type": "concept", "mode": "whitebox", "source": "§1", "mastery": 0.5},
            "prerequisites": [],
            "related": [],
        }
        questions = client.generate_questions(ctx, n_questions=2)
        assert len(questions) >= 1
        assert len(questions) <= 2

    def test_evaluate_answer(self, client: LLMClient) -> None:
        """evaluate_answer() 返回 (score, feedback)。"""
        mock_choice = MagicMock()
        mock_choice.message.content = '{"score": 1.0, "feedback": "Great answer!"}'
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client

        ctx = {"item": {"title": "CLT"}}
        score, feedback = client.evaluate_answer("What is CLT?", "sample mean distribution", "The CLT states...", ctx)
        assert score == 1.0
        assert "Great" in feedback


# ── _parse_score_response 测试 ────────────────────────────────────────


class TestParseScoreResponse:
    """_parse_score_response() 测试。"""

    def test_plain_json(self) -> None:
        score, fb = _parse_score_response('{"score": 1.0, "feedback": "correct"}')
        assert score == 1.0
        assert fb == "correct"

    def test_markdown_code_block(self) -> None:
        response = '```json\n{"score": 0.5, "feedback": "partial"}\n```'
        score, fb = _parse_score_response(response)
        assert score == 0.5
        assert fb == "partial"

    def test_invalid_json_fallback(self) -> None:
        score, fb = _parse_score_response("not json at all")
        assert score == 0.5
        assert fb == "not json at all"

    def test_score_clamped(self) -> None:
        score, _ = _parse_score_response('{"score": 2.0, "feedback": "x"}')
        assert score == 1.0


# ── 便捷函数测试 ──────────────────────────────────────────────────────


class TestIsLLMAvailable:
    """is_llm_available() 测试。"""

    def test_no_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert not is_llm_available()

    def test_with_key(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-test"}, clear=True):
            assert is_llm_available()

    def test_ollama(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            assert is_llm_available()


class TestGetConfigSummary:
    """get_config_summary() 测试。"""

    def test_returns_dict(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-verylongkey12345"}, clear=True):
            summary = get_config_summary()
            assert summary["provider"] == "deepseek"
            assert summary["model"] == "deepseek-chat"
            assert "..." in summary["api_key_masked"]
            assert summary["available"] == "是"

    def test_no_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            summary = get_config_summary()
            assert "(未设置)" in summary["api_key_masked"]
            assert summary["available"] == "否"
