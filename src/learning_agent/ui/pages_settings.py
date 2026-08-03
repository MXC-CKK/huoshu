"""Streamlit 模型设置页。

提供直观的窗口选择/配置 LLM：
- 提供商选择（DeepSeek / OpenAI / Ollama）
- API Key 输入（密码框）
- Base URL / 模型名自动补全，可手动覆盖
- 测试连接
- 保存到 ~/.huoshu/config.json（仅当前用户可读）

用法:
    streamlit run src/learning_agent/ui/pages_settings.py
"""

from __future__ import annotations

import importlib
import importlib.util

if importlib.util.find_spec("streamlit"):
    import streamlit as st
else:  # pragma: no cover - 未安装 UI 依赖时的降级
    st = None  # type: ignore[assignment]

from learning_agent.llm import (
    DEFAULT_DEEPSEEK_BASE,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_OLLAMA_BASE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_BASE,
    DEFAULT_OPENAI_MODEL,
    LLMClient,
    LLMConfig,
)


def _bad_base_url_reason(provider: str, base_url: str) -> str:
    """检查 Base URL 是否与客户端协议不匹配，返回原因（空串=合法）。

    客户端使用 OpenAI 兼容协议（/v1/chat/completions）。
    若填了 Anthropic 兼容端点（如 …/anthropic）则无法工作。
    """
    if not base_url:
        return "Base URL 不能为空"
    if provider != "ollama" and "/anthropic" in base_url:
        return (
            f"{base_url} 是 Anthropic 兼容端点，本工具使用 OpenAI 兼容协议。\n"
            f"DeepSeek 请用 https://api.deepseek.com/v1，OpenAI 请用 https://api.openai.com/v1"
        )
    if provider == "ollama" and "11434" not in base_url:
        return f"Ollama 默认地址应为 http://localhost:11434/v1，当前: {base_url}"
    return ""


def _warn_bad_base_url(provider: str, base_url: str) -> None:
    """在表单下方显示 Base URL 合法性警告（不阻塞输入）。"""
    reason = _bad_base_url_reason(provider, base_url.strip())
    if reason:
        st.warning(f"⚠️ {reason}")


# 提供商 → (默认 base_url, 默认 model, 提示文案)
PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": DEFAULT_DEEPSEEK_BASE,
        "model": DEFAULT_DEEPSEEK_MODEL,
        "hint": "DeepSeek 官方 API（国内直连，性价比高）",
        "api_key_hint": "sk-...（https://platform.deepseek.com 获取）",
    },
    "openai": {
        "base_url": DEFAULT_OPENAI_BASE,
        "model": DEFAULT_OPENAI_MODEL,
        "hint": "OpenAI 官方 API（需海外网络）",
        "api_key_hint": "sk-...（https://platform.openai.com 获取）",
    },
    "ollama": {
        "base_url": DEFAULT_OLLAMA_BASE,
        "model": DEFAULT_OLLAMA_MODEL,
        "hint": "本地 Ollama（免费，无需 API Key）",
        "api_key_hint": "本地模型无需 Key，留空即可",
    },
}


def main() -> None:
    """模型设置页入口。"""
    if st is None:
        print("Streamlit 未安装。请运行: pip install streamlit")
        raise SystemExit(1)

    st.set_page_config(
        page_title="模型设置 · 活书",
        page_icon="⚙️",
        layout="wide",
    )
    st.title("⚙️ 模型设置")
    st.caption("配置学习会话使用的 LLM。保存后立即生效，配置存储在本地，不会上传。")

    # 当前生效配置（文件 > 环境变量）
    current = LLMConfig.resolve()

    # ── 表单 ──
    with st.form("llm_config_form", clear_on_submit=False):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📦 提供商")
            provider = st.radio(
                "选择模型提供商",
                options=list(PROVIDERS.keys()),
                format_func=lambda p: {
                    "deepseek": "🟣 DeepSeek",
                    "openai": "⚪ OpenAI",
                    "ollama": "🟠 Ollama (本地)",
                }.get(p, p),
                index=list(PROVIDERS.keys()).index(
                    current.provider if current.provider in PROVIDERS else "deepseek"
                ),
                help="Ollama 为本地模型，无需 API Key，适合隐私敏感场景",
            )
            st.caption(PROVIDERS[provider]["hint"])

        with col2:
            st.subheader("🔑 连接配置")

            # 当前选中 provider 的推荐值（跟随 radio 选择）
            default_base = PROVIDERS[provider]["base_url"]
            default_model = PROVIDERS[provider]["model"]
            # 配置文件中的值（如有）——仅在 provider 匹配时回填
            file_cfg_now = LLMConfig.from_file()
            if file_cfg_now is not None and file_cfg_now.provider == provider:
                suggested_base = file_cfg_now.base_url or default_base
                suggested_model = file_cfg_now.model or default_model
                suggested_key = file_cfg_now.api_key
            else:
                suggested_base = default_base
                suggested_model = default_model
                suggested_key = ""

            # API Key（密码框）
            api_key = st.text_input(
                "API Key",
                type="password",
                value=suggested_key,
                placeholder=PROVIDERS[provider]["api_key_hint"],
                help="密钥只保存在本机 ~/.huoshu/config.json（权限 600），不会上传或提交",
            )

            # Base URL（跟随当前提供商默认值）
            base_url = st.text_input(
                "Base URL",
                value=suggested_base,
                placeholder=default_base,
                help="API 端点地址。支持自定义中转/代理地址",
            )
            # URL 合法性提示（防填错端点格式）
            _warn_bad_base_url(provider, base_url)

            # 模型名（跟随当前提供商默认值）
            model = st.text_input(
                "模型名",
                value=suggested_model,
                placeholder=default_model,
                help="如 deepseek-chat / gpt-4o-mini / llama3.2（Ollama 需先 pull）",
            )

            # 高级参数
            with st.expander("⚡ 高级参数"):
                temperature = st.slider(
                    "Temperature（生成随机性）",
                    min_value=0.0,
                    max_value=1.5,
                    value=float(current.temperature),
                    step=0.1,
                    help="越低越确定，越高越有创造性。教学场景推荐 0.5–0.8",
                )
                max_tokens = st.number_input(
                    "Max Tokens（最大输出长度）",
                    min_value=128,
                    max_value=8192,
                    value=int(current.max_tokens),
                    step=128,
                )
                proxy = st.text_input(
                    "代理（Proxy，可选）",
                    value=current.proxy,
                    placeholder="http://127.0.0.1:7890",
                    help="网络无法直连 API 时填写代理地址（如公司代理/科学上网）。留空则使用系统默认网络",
                )

        st.divider()
        col_save, col_test, _ = st.columns([1, 1, 2])
        with col_save:
            submitted = st.form_submit_button("💾 保存配置", type="primary", use_container_width=True)
        with col_test:
            test_clicked = st.form_submit_button("🔌 测试连接", use_container_width=True)

    # ── 保存 ──
    if submitted:
        # 保存前校验 Base URL
        bad_url = _bad_base_url_reason(provider, base_url.strip())
        if bad_url:
            st.error(f"❌ 无法保存：{bad_url}")
        else:
            cfg = LLMConfig.from_dict({
                "provider": provider,
                "api_key": api_key.strip(),
                "base_url": base_url.strip(),
                "model": model.strip(),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "proxy": proxy.strip(),
            })
            # Ollama 不需要 key
            if provider == "ollama":
                cfg.api_key = ""
            path = cfg.save()
            st.success(f"✅ 配置已保存到 {path}")
            st.session_state["llm_config"] = cfg.to_dict()
            st.rerun()

    # ── 测试连接 ──
    if test_clicked:
        bad_url = _bad_base_url_reason(provider, base_url.strip())
        if bad_url:
            st.error(f"❌ 无法测试：{bad_url}")
        else:
            test_cfg = LLMConfig.from_dict({
                "provider": provider,
                "api_key": api_key.strip(),
                "base_url": base_url.strip(),
                "model": model.strip(),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "proxy": proxy.strip(),
            })
            if provider == "ollama":
                test_cfg.api_key = ""
            with st.spinner("正在测试连接..."):
                try:
                    client = LLMClient(config=test_cfg)
                    reply = client.chat(
                        [{"role": "user", "content": "你好！请只回复：连接成功"}],
                        max_tokens=20,
                    )
                    st.success(f"✅ 连接成功！模型回复：{reply[:60]}")
                except Exception as exc:  # noqa: BLE001 - 连接失败需兜底展示给用户
                    st.error(f"❌ 连接失败：{exc}")

    # ── 当前状态 ──
    st.divider()
    st.subheader("📋 当前配置")
    file_cfg = LLMConfig.from_file()
    env_cfg = LLMConfig.from_env()

    from streamlit_shadcn_ui import alert

    col_a, col_b = st.columns(2)
    with col_a:
        if file_cfg is not None:
            alert(
                title="生效来源：配置文件",
                description=(
                    f"~/.huoshu/config.json — "
                    f"`{file_cfg.provider}` / `{file_cfg.model}`"
                    f" · API Key: {'已设置' if file_cfg.api_key else '未设置'}"
                ),
                variant="default",
                key="cfg_alert",
            )
        else:
            alert(
                title="生效来源：环境变量 / 默认值",
                description=(
                    f"`{env_cfg.provider}` / `{env_cfg.model}`"
                    f" · API Key: {'已设置' if env_cfg.api_key else '未设置（学习会话将使用模板回复）'}"
                ),
                variant="default",
                key="cfg_alert",
            )
    with col_b:
        st.caption("🔒 隐私说明")
        st.caption("• API Key 仅保存在本机，权限 600（仅你可见）")
        st.caption("• 配置文件在 ~/.huoshu/，不会进入项目 git")
        st.caption("• 不保存时仅本次会话生效（环境变量方式）")


if __name__ == "__main__":
    main()
