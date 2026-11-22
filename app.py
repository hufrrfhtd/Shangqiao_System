from base64 import b64encode

import streamlit as st

from llm import deepseek_enabled
from settings import PROJECT_DIR

st.set_page_config(
    page_title="耗材工作台 · AI-Interview-Demo",
    page_icon="🖨️",
    layout="wide",
)

def _module_sidebar() -> str:
    with st.sidebar:
        st.html(
            '<div class="workspace-brand"><span class="workspace-brand-mark" aria-hidden="true">AI</span>'
            '<span>耗材工作台</span></div>'
        )
        selected = st.radio(
            "业务模块",
            ("业务助手", "商品智能分类"),
            key="workspace_module",
            width="stretch",
            on_change=st.session_state.update,
            kwargs={"business_view": "home"},
        )
        connected = deepseek_enabled()
        connection = "DeepSeek 已配置" if connected else "本地模式"
        mode = "connected" if connected else "local"
        st.html(
            f'<div class="workspace-status"><span class="workspace-status-dot {mode}"></span>'
            f'<span>{connection}</span></div><p class="workspace-demo-note">业务数据为虚构样例</p>'
        )
    return selected


def _business_home() -> None:
    with st.container(key="business_home"):
        st.html(
            '<header class="business-home-title"><h1>业务助手</h1>'
            '<p>打印机耗材的询价、库存与售后</p></header>'
        )
        quote, chat = st.columns(2, gap="medium")
        with quote, st.container(border=True, key="entry_quote", height="stretch"):
            st.markdown(":material/request_quote:")
            st.subheader("询价报价", anchor=False)
            st.write("上传 Excel、CSV 或直接输入询价，审核后导出报价草稿。")
            st.button(
                "进入询价", key="open_quote", type="primary", icon=":material/arrow_forward:",
                width="stretch", on_click=st.session_state.update, kwargs={"business_view": "quote"},
            )
        with chat, st.container(border=True, key="entry_chat", height="stretch"):
            st.markdown(":material/forum:")
            st.subheader("自然语言对话", anchor=False)
            st.write("直接提问，查询库存、价格、耗材兼容性与设备故障。")
            st.button(
                "开始对话", key="open_chat", icon=":material/arrow_forward:", width="stretch",
                on_click=st.session_state.update, kwargs={"business_view": "chat"},
            )


def _classification_hero() -> None:
    cover = PROJECT_DIR / "pages_ui" / "assets" / "toner-workspace.webp"
    cover_img = ""
    if cover.is_file():
        image_data = b64encode(cover.read_bytes()).decode("ascii")
        cover_img = f'<img class="workspace-hero-photo" src="data:image/webp;base64,{image_data}" alt="">'
    st.html(
        '<div class="workspace-hero">'
        f'{cover_img}<div class="workspace-hero-copy"><h1>商品智能分类</h1>'
        '<p>批量预测耗材类目，复核后导出带标准 ERP 编码的商品清单。</p></div></div>'
    )


st.html(PROJECT_DIR / "pages_ui" / "workspace.css")
view = st.session_state.get("business_view", "home")

if view == "chat":
    from pages_ui.rag_page import render

    render()
else:
    selected = _module_sidebar()
    if selected == "商品智能分类":
        from pages_ui.classification_page import render

        _classification_hero()
        render()
    elif view == "quote":
        from pages_ui.quote_page import render

        st.button(
            "返回业务助手", key="back_to_business_quote", type="tertiary", icon=":material/arrow_back:",
            on_click=st.session_state.update, kwargs={"business_view": "home"},
        )
        st.html(
            '<header class="business-quote-title"><h1>询价报价</h1>'
            '<p>对照两路抽取和 ERP 核验结果，审核后生成报价草稿。</p></header>'
        )
        render()
    else:
        _business_home()

