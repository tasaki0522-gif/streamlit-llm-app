
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# --- .env と Streamlit Secrets からAPIキーを読む ---
load_dotenv()  # ローカル用（.env）

# 1. .env から読む
api_key_env = os.getenv("OPENAI_API_KEY")
# 2. なければ Streamlit Secrets から読む（本番用）
OPENAI_API_KEY = api_key_env or st.secrets.get("OPENAI_API_KEY")

# キーが無い場合は画面にエラーを出す
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が設定されていません。Streamlitの Secrets または .env を確認してください。")


# --- LLMに質問を送る関数 ---
def ask_expert(user_input: str, choice: str) -> str:
    """
    入力テキスト（user_input）と
    ラジオボタンで選んだ値（choice: 'A' or 'B'）を受け取り、
    LangChain経由でLLMの回答テキストを返す
    """

    # choice に応じてシステムメッセージを切り替え
    if choice == "A":
        # A：勉強法・学習計画の専門家
        system_prompt = (
            "あなたは勉強法と学習計画の専門家です。"
            "中学生にも分かる言葉で、具体的なやり方やステップを示しながら、"
            "優しく丁寧にアドバイスしてください。"
        )
    elif choice == "B":
        # B：メンタル・モチベーションの専門家
        system_prompt = (
            "あなたはメンタルケアとモチベーションの専門家です。"
            "相手の気持ちに寄りそいながら、不安や悩みを整理し、"
            "前向きになれる具体的な行動を提案してください。"
        )
    else:
        # 想定外の場合の保険
        system_prompt = "あなたは親切で丁寧なアシスタントです。"

    chat = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    response = chat.invoke(messages)
    return response.content


# --- Streamlitアプリのメイン部分 ---
def main():
    st.title("💬 LLM専門家チャットアプリ（A/B 切り替え）")
    st.write("このアプリでは、2種類の専門家に質問できます。")
    st.write("AまたはBを選び、質問を入力して『送信』ボタンを押してください。")
    st.write("A：勉強法・学習計画の専門家 / B：メンタル・モチベーションの専門家")

    # APIキーがないと動かないのでここで止める
    if not OPENAI_API_KEY:
        st.stop()

    # --- ラジオボタンで A / B を選択 ---
    # 表示ラベルと内部の値を分ける
    label_to_choice = {
        "A：勉強法・学習計画の専門家": "A",
        "B：メンタル・モチベーションの専門家": "B",
    }

    selected_label = st.radio(
        "どちらの専門家に相談しますか？",
        list(label_to_choice.keys()),
    )
    choice = label_to_choice[selected_label]  # 'A' または 'B'

    # --- 入力フォーム（1つ） ---
    user_input = st.text_input("質問や相談内容を入力してください：")

    # --- ボタンが押されたときの処理 ---
    if st.button("送信"):
        if not user_input.strip():
            st.warning("質問を入力してください。")
        else:
            with st.spinner("AIが考え中です..."):
                answer = ask_expert(user_input, choice)
                st.subheader("🔎 回答：")
                st.write(answer)


if __name__ == "__main__":
    main()

