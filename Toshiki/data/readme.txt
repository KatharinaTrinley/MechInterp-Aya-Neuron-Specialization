cloze and translation datasets are from (Dumas et al., 2025) repository
https://anonymous.4open.science/r/llm-lang-agnostic-1A75/README.md

the following code was used for preprocessing (dumas et al., 2025) datasets
https://colab.research.google.com/drive/17gr0lcmMH9ndqOdVWScRq3NCn5rdTqjA?usp=sharing

ko-ja_translation.json was generated using the following code
https://colab.research.google.com/drive/1mOdvRhTITPdGEI--e61lIi8zidtwvi9O?usp=sharing

data preprocessing
https://colab.research.google.com/drive/18QnNZ6gkon0VofFdA7h-Q7x_rRLEOlZj?usp=sharing

prompt given to chatgpt to generate translation prompts
あなたはLLMのプロンプトを考える専門家です。下記の２つの文は、LLMにおいて翻訳タスクを行うためのプロンプトです。１と２に対してそれぞれ同等の意味を持つプロンプトを、日本語と韓国語でそれぞれ１０個ずつ生成してください。
プロンプト１：次の日本語を韓国語に翻訳してください。
プロンプト２：次の韓国語を日本語に翻訳してください。


プロンプト1（日本語 → 韓国語）
日本語（10個）

    次の文章を韓国語に訳してください。
    以下の日本語を韓国語に翻訳してください。
    次のテキストを韓国語に変換してください。
    下記の文を韓国語に直してください。
    この日本語を韓国語に訳してもらえますか？
    次の日本語の文章を韓国語に翻訳してほしいです。
    こちらの文を韓国語に訳してください。
    次の内容を韓国語にしてください。
    下の日本語を韓国語に言い換えてください。
    この日本語を韓国語に翻訳するとどうなりますか？

韓国語（10個）

    다음 문장을 한국어로 번역해주세요.
    아래 일본어를 한국어로 번역해주세요.
    다음 텍스트를 한국어로 변환해주세요.
    아래 문장을 한국어로 고쳐주세요.
    이 일본어를 한국어로 번역해 줄 수 있나요?
    다음 일본어 문장을 한국어로 번역해 주세요.
    이 문장을 한국어로 바꿔주세요.
    다음 내용을 한국어로 표현해주세요.
    아래 일본어를 한국어로 바꿔 주세요.
    이 일본어 문장을 한국어로 번역하면 어떻게 되나요?

プロンプト2（韓国語 → 日本語）
日本語（10個）

    次の韓国語を日本語に訳してください。
    以下の韓国語を日本語に翻訳してください。
    次の文章を日本語に変換してください。
    下記の文を日本語に直してください。
    この韓国語を日本語に訳してもらえますか？
    次の韓国語の文章を日本語に翻訳してほしいです。
    こちらの文を日本語に訳してください。
    次の内容を日本語にしてください。
    下の韓国語を日本語に言い換えてください。
    この韓国語を日本語に翻訳するとどうなりますか？

韓国語（10個）

    다음 한국어 문장을 일본어로 번역해주세요.
    아래 한국어를 일본어로 번역해주세요.
    다음 텍스트를 일본어로 변환해주세요.
    아래 문장을 일본어로 고쳐주세요.
    이 한국어를 일본어로 번역해 줄 수 있나요?
    다음 한국어 문장을 일본어로 번역해 주세요.
    이 문장을 일본어로 바꿔주세요.
    다음 내용을 일본어로 표현해주세요.
    아래 한국어를 일본어로 바꿔 주세요.
    이 한국어 문장을 일본어로 번역하면 어떻게 되나요?