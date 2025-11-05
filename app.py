# app.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# --- .envファイルの読み込み ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLMに質問を送る関数 ---
def ask_expert(user_input, expert_type):
    """
    入力テキスト（user_input）と専門家タイプ（expert_type）を受け取り、
    LangChain経由でLLMの回答を返す関数
    """
    # 専門家のタイプに応じてシステムメッセージを変更
    system_prompt = f"あなたは{expert_type}の専門家です。わかりやすく、丁寧に説明してください。"

    chat = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    response = chat.invoke(messages)
    return response.content


# --- Streamlitアプリのメイン部分 ---
def main():
    st.title("💬 LLM専門家チャットアプリ")
    st.write("このアプリでは、専門家の視点から質問に答えてくれるAIと会話できる。")
    st.write("1️⃣ 専門家のタイプをラジオボタンで選ぶ。")
    st.write("2️⃣ 下の入力欄に質問を入力する。")
    st.write("3️⃣ 『送信』ボタンを押すと、選んだ専門家が回答する。")

    # --- ラジオボタンで専門家タイプを選択 ---
    expert_type = st.radio(
        "専門家のタイプを選んでください：",
        ["科学者", "心理カウンセラー", "歴史学者", "プログラマー"]
    )

    # --- 入力フォーム ---
    user_input = st.text_input("質問を入力してください：")

    # --- ボタンが押されたときの処理 ---
    if st.button("送信"):
        if not user_input.strip():
            st.warning("質問を入力してください。")
        else:
            with st.spinner("AIが考え中です..."):
                answer = ask_expert(user_input, expert_type)
                st.subheader("🔎 回答：")
                st.write(answer)


if __name__ == "__main__":
    main()
