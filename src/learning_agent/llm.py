"""LLM 客户端 — DeepSeek / OpenAI / Ollama 统一接口。

通过环境变量配置 Provider，支持热切换：
    LLM_PROVIDER=deepseek  (默认)
    LLM_API_KEY=sk-...
    LLM_BASE_URL=https://api.deepseek.com/v1
    LLM_MODEL=deepseek-chat

也支持通过 Python API 直接传参覆盖:
    from learning_agent.llm import LLMClient

    client = LLMClient.from_env()
    reply = client.chat("用 Socratic 方法解释大数定律")

典型用法:
    client = LLMClient(api_key="sk-...", base_url="...", model="deepseek-chat")
    reply = client.socratic_teach(item_context, user_question, chat_history)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 默认配置 ─────────────────────────────────────────────────────────

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.2"

# 配置文件路径（用户级，不入库）
DEFAULT_CONFIG_DIR = Path.home() / ".huoshu"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"


# ── 配置 ─────────────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    """LLM 连接配置。

    Attributes:
        provider: Provider 标识（deepseek/openai/ollama）。
        api_key: API 密钥。
        base_url: API base URL。
        model: 模型名。
        temperature: 生成温度（0=确定性, 1=创造性）。
        max_tokens: 最大输出 token 数。
    """

    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = DEFAULT_DEEPSEEK_BASE
    model: str = DEFAULT_DEEPSEEK_MODEL
    temperature: float = 0.7
    max_tokens: int = 1024
    proxy: str = ""

    @classmethod
    def from_env(cls) -> LLMConfig:
        """从环境变量加载配置。

        环境变量:
            LLM_PROVIDER: deepseek / openai / ollama（默认 deepseek）。
            LLM_API_KEY: API 密钥。
            LLM_BASE_URL: API base URL（可选，按 provider 自动补默认值）。
            LLM_MODEL: 模型名（可选，按 provider 自动补默认值）。

        Returns:
            LLMConfig 实例。
        """
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        api_key = os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "")
        model = os.getenv("LLM_MODEL", "")
        proxy = os.getenv("LLM_PROXY", "")

        # 按 provider 补默认值
        if not base_url:
            base_url = {
                "deepseek": DEFAULT_DEEPSEEK_BASE,
                "openai": DEFAULT_OPENAI_BASE,
                "ollama": DEFAULT_OLLAMA_BASE,
            }.get(provider, DEFAULT_DEEPSEEK_BASE)

        if not model:
            model = {
                "deepseek": DEFAULT_DEEPSEEK_MODEL,
                "openai": DEFAULT_OPENAI_MODEL,
                "ollama": DEFAULT_OLLAMA_MODEL,
            }.get(provider, DEFAULT_DEEPSEEK_MODEL)

        # 防护：环境变量可能残留其他 provider 的值（如 LLM_MODEL=ollama/...）
        # 当 provider 与 model/base_url 不匹配时，忽略环境变量用默认值
        if provider != "ollama" and model.startswith("ollama"):
            logger.warning(
                "LLM_MODEL='%s' 与 provider='%s' 不匹配，忽略环境变量",
                model, provider,
            )
            model = DEFAULT_DEEPSEEK_MODEL if provider == "deepseek" else DEFAULT_OPENAI_MODEL
        if provider != "ollama" and "localhost:11434" in base_url:
            logger.warning(
                "LLM_BASE_URL='%s' 与 provider='%s' 不匹配，忽略环境变量",
                base_url, provider,
            )
            base_url = DEFAULT_DEEPSEEK_BASE if provider == "deepseek" else DEFAULT_OPENAI_BASE

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            proxy=proxy,
        )

    # ── 配置文件支持（UI 设置页持久化）──

    @classmethod
    def from_file(cls, path: Path | None = None) -> LLMConfig | None:
        """从配置文件加载（~/.huoshu/config.json）。

        Args:
            path: 配置文件路径，默认 ~/.huoshu/config.json。

        Returns:
            LLMConfig 实例；文件不存在或解析失败时返回 None。
        """
        cfg_path = path or DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            return None
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("读取 LLM 配置失败: %s (%s)", cfg_path, exc)
            return None
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        """从字典构建配置（UI 表单提交）。

        Args:
            data: 配置字典（provider/api_key/base_url/model 等）。

        Returns:
            LLMConfig 实例。
        """
        provider = str(data.get("provider", "deepseek")).lower()
        api_key = str(data.get("api_key", ""))
        base_url = str(data.get("base_url", ""))
        model = str(data.get("model", ""))

        defaults = _provider_defaults(provider)
        if not base_url:
            base_url = defaults["base_url"]
        if not model:
            model = defaults["model"]

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=float(data.get("temperature", 0.7)),
            max_tokens=int(data.get("max_tokens", 1024)),
            proxy=str(data.get("proxy", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """导出为字典（UI 表单回填 / 持久化）。"""
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "proxy": self.proxy,
        }

    def save(self, path: Path | None = None) -> Path:
        """保存配置到文件（~/.huoshu/config.json）。

        Args:
            path: 保存路径，默认 ~/.huoshu/config.json。

        Returns:
            保存的文件路径。
        """
        cfg_path = path or DEFAULT_CONFIG_PATH
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            os.chmod(cfg_path, 0o600)  # 仅当前用户可读写（含 API key）
        except OSError:
            pass
        logger.info("LLM 配置已保存: %s", cfg_path)
        return cfg_path

    @staticmethod
    def resolve() -> LLMConfig:
        """解析最终生效配置：文件 > 环境变量 > 默认值。

        Returns:
            生效的 LLMConfig 实例。
        """
        file_cfg = LLMConfig.from_file()
        if file_cfg is not None and file_cfg.api_key:
            return file_cfg
        return LLMConfig.from_env()


def _provider_defaults(provider: str) -> dict[str, str]:
    """返回 provider 的默认 base_url 和 model。"""
    return {
        "base_url": {
            "deepseek": DEFAULT_DEEPSEEK_BASE,
            "openai": DEFAULT_OPENAI_BASE,
            "ollama": DEFAULT_OLLAMA_BASE,
        }.get(provider, DEFAULT_DEEPSEEK_BASE),
        "model": {
            "deepseek": DEFAULT_DEEPSEEK_MODEL,
            "openai": DEFAULT_OPENAI_MODEL,
            "ollama": DEFAULT_OLLAMA_MODEL,
        }.get(provider, DEFAULT_DEEPSEEK_MODEL),
    }


# ── 客户端 ───────────────────────────────────────────────────────────


@dataclass
class LLMClient:
    """LLM 调用客户端 — 封装 OpenAI 兼容 API。

    DeepSeek 和 Ollama 都兼容 OpenAI chat/completions 协议，
    通过 base_url 区分 Provider。

    Attributes:
        config: LLMConfig 配置。
    """

    config: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_env(cls) -> LLMClient:
        """创建 LLMClient（配置文件 > 环境变量 > 默认值）。

        优先使用 UI 设置页保存的配置（~/.huoshu/config.json），
        未配置时回退到环境变量。
        """
        return cls(config=LLMConfig.resolve())

    @property
    def available(self) -> bool:
        """LLM 是否可用（有 API key 或为 ollama 本地模式）。"""
        if self.config.provider == "ollama":
            return True  # ollama 本地不需要 key
        return bool(self.config.api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """发送多轮对话，返回模型回复文本。

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 生成温度（None 用 config 默认值）。
            max_tokens: 最大输出 token（None 用 config 默认值）。

        Returns:
            模型回复文本。

        Raises:
            RuntimeError: LLM 不可用或调用失败。
        """
        if not self.available:
            raise RuntimeError(
                "LLM 不可用。请设置环境变量 LLM_API_KEY（或 LLM_PROVIDER=ollama 使用本地模型）。"
            )

        import openai

        client = openai.OpenAI(
            api_key=self.config.api_key or "ollama",  # ollama 不需要真实 key
            base_url=self.config.base_url,
            http_client=(
                openai.DefaultHttpxClient(proxy=self.config.proxy)
                if self.config.proxy
                else None
            ),
        )

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            )
            content = response.choices[0].message.content
            return content if content else ""

        except openai.AuthenticationError as exc:
            raise RuntimeError(
                f"LLM 认证失败。请检查 LLM_API_KEY。Provider: {self.config.provider}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise RuntimeError(
                f"LLM 连接失败。请检查 LLM_BASE_URL ({self.config.base_url}) 和网络。"
            ) from exc
        except Exception as exc:
            logger.exception("LLM 调用异常")
            raise RuntimeError(f"LLM 调用失败: {exc}") from exc

    # ── 高级接口 ─────────────────────────────────────────────────

    def socratic_teach(
        self,
        item_context: dict[str, Any],
        user_question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Socratic 教学：基于图谱上下文引导式解答用户问题。

        Args:
            item_context: build_llm_context() 的返回字典。
            user_question: 用户当前问题。
            chat_history: 之前的对话历史。

        Returns:
            模型的 Socratic 引导回复。
        """
        system_prompt = _build_socratic_system_prompt(item_context)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": user_question})

        logger.info(
            "Socratic teach: item=%s, qtype=%s",
            item_context.get("item", {}).get("id", "?"),
            item_context.get("instruction", "")[:50],
        )
        return self.chat(messages)

    def generate_questions(
        self,
        item_context: dict[str, Any],
        n_questions: int = 3,
    ) -> list[str]:
        """基于知识点生成自适应测试题。

        Args:
            item_context: build_llm_context() 的返回字典。
            n_questions: 出题数量。

        Returns:
            题目文本列表。
        """
        item = item_context.get("item", {})
        mastery = item.get("mastery", 0.5)
        difficulty = "基础" if mastery < 0.4 else "理解" if mastery < 0.7 else "应用"

        prompt = (
            f"请为知识点「{item.get('title', '')}」出 {n_questions} 道{difficulty}难度的测试题。\n"
            f"教材锚点: {item.get('source', '')}\n"
            f"要求: 每题独立、可检验理解、覆盖不同角度。"
        )

        response = self.chat([
            {"role": "system", "content": "你是一位严谨的学科教师。请只输出题目，每题一行，用数字序号开头。"},
            {"role": "user", "content": prompt},
        ])

        # 按序号拆分
        import re
        questions = re.split(r"\n(?=\d+[\.、)）])", response.strip())
        return [q.strip() for q in questions if q.strip()][:n_questions]

    def evaluate_answer(
        self,
        question: str,
        correct_answer_hint: str,
        user_answer: str,
        item_context: dict[str, Any],
    ) -> tuple[float, str]:
        """让 LLM 评判答案（替代关键词匹配）。

        Args:
            question: 题目文本。
            correct_answer_hint: 参考答案提示。
            user_answer: 用户答案。
            item_context: 知识点上下文。

        Returns:
            (score, feedback) — score ∈ {0.0, 0.5, 1.0}, feedback 是文本。
        """
        item = item_context.get("item", {})
        prompt = (
            f"知识点: {item.get('title', '')}\n"
            f"题目: {question}\n"
            f"参考答案要点: {correct_answer_hint}\n"
            f"学生答案: {user_answer}\n\n"
            f"请评判这个答案。返回 JSON: {{\"score\": 1.0|0.5|0.0, \"feedback\": \"...\"}}\n"
            f"1.0=完全正确或核心理解到位, 0.5=部分正确, 0.0=错误或完全偏题。"
        )

        response = self.chat([
            {"role": "system", "content": "你是一位严谨的评分教师。请只返回 JSON，不要其他内容。"},
            {"role": "user", "content": prompt},
        ])

        return _parse_score_response(response)


# ── system prompt 构建 ────────────────────────────────────────────────


def _build_socratic_system_prompt(ctx: dict[str, Any]) -> str:
    """按 learn-session 协议构建 Socratic 教学 system prompt。

    包含: 图谱位置、前置知识、相关概念、教学指令、防幻觉规则。
    """
    item = ctx.get("item", {})
    prerequisites = ctx.get("prerequisites", [])
    related = ctx.get("related", [])
    instruction = ctx.get("instruction", "")

    lines = [
        "你是一位专业的学科教师，遵循 Socratic 教学法。",
        "",
        "## 当前知识点",
        f"- 标题: {item.get('title', '')}",
        f"- 类型: {item.get('type', '')}",
        f"- 模式: {item.get('mode', '')}（{'深入理解，需要解释证明动机、出变式题' if item.get('mode') == 'whitebox' else '会用即可，只讲用法和公式形态，不深入证明'}）",
        f"- 掌握度: {item.get('mastery', 0):.0%}",
        f"- 教材锚点: {item.get('source', '')}",
    ]

    if item.get("note"):
        lines.append(f"- 要点: {item.get('note')}")

    if prerequisites:
        lines.append("")
        lines.append("## 前置知识（用户已学或正在补）")
        for p in prerequisites:
            lines.append(f"- {p.get('title', '')} (掌握度: {p.get('mastery', 0):.0%})")

    if related:
        lines.append("")
        lines.append("## 相关概念")
        for r in related:
            lines.append(f"- {r.get('title', '')}")

    lines.append("")
    lines.append("## 教学指令")
    lines.append(instruction)

    lines.append("")
    lines.append("## 核心规则（必须遵守）")
    lines.append("1. **Socratic 优先**: 先给引导性追问，不要直接给答案。用户表示不理解时再逐步解释。")
    lines.append("2. **教材锚点**: 所有定理/公式/定义引用必须标注教材锚点（source 字段）。")
    lines.append("3. **钉回图谱**: 解释完成后明确标注「这条解释对应节点 {item_id}」。")
    lines.append("4. **防幻觉**: 对教材内容不确定时明确说「待查证」，不要凭空编造。")
    lines.append("5. **黑箱克制**: 如果 mode=blackbox，只讲使用条件和公式形态，不讲证明过程。")
    lines.append("6. **术语翻译**: 对用户永远不说「黑箱/白箱/掌握度/节点 id」这些内部术语。")
    lines.append("7. **简短有力**: 每次回复控制在 3-5 句话，引导而非灌输。")

    return "\n".join(lines)


def _parse_score_response(response: str) -> tuple[float, str]:
    """解析 LLM 返回的评分 JSON。

    Args:
        response: LLM 原始响应（可能包含 markdown 代码块）。

    Returns:
        (score, feedback) — 解析失败时默认 (0.5, 原始响应)。
    """
    import json

    # 尝试提取 JSON 块
    json_str = response
    if "```json" in response:
        match = response.split("```json")[1].split("```")[0]
        json_str = match.strip()
    elif "```" in response:
        match = response.split("```")[1].split("```")[0]
        json_str = match.strip()

    try:
        data = json.loads(json_str)
        score = float(data.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        feedback = str(data.get("feedback", ""))
        return score, feedback
    except (json.JSONDecodeError, ValueError, KeyError):
        logger.debug("LLM score response parse failed, using raw response")
        return 0.5, response


# ── 便捷函数 ─────────────────────────────────────────────────────────


def is_llm_available() -> bool:
    """检查 LLM 是否可用（API key 已设或 Ollama 本地）。

    优先读取配置文件（~/.huoshu/config.json），其次环境变量。
    """
    config = LLMConfig.resolve()
    if config.provider == "ollama":
        return True
    return bool(config.api_key)


def get_config_summary() -> dict[str, str]:
    """获取当前 LLM 配置摘要（不暴露完整 API key）。

    Returns:
        含 provider, model, base_url, key_masked 的字典。
    """
    config = LLMConfig.from_env()
    key = config.api_key
    masked = key[:8] + "..." + key[-4:] if len(key) > 12 else ("***" if key else "(未设置)")
    return {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_masked": masked,
        "available": "是" if LLMClient(config=config).available else "否",
    }
