from fish_data_engine.annotation.api import (
    init_database,
    login_user,
    get_new_sample,
    update_sample,
)
import gradio as gr

init_database()

HEADER_MD = """
# Fish Data

数据标注工具.
"""


def call_login_user(username, password):
    user = login_user(username, password)
    if user is None:
        return "用户名或密码错误", None, None

    sample = get_new_sample("audio-quality", user["_id"])
    if sample is None:
        return "没有待标注的样本", None, None

    return "已获取样本", sample, sample["data"]


def call_submit_sample(username, password, sample, audio, clear, checkboxs):
    user = login_user(username, password)
    if user is None:
        return "用户名或密码错误", sample, audio

    if sample is None:
        return "无工作中的样本", sample, audio

    if clear is None:
        return "请选择音频质量", sample, audio

    if "保留样本" in checkboxs and not ("纯人声" in checkboxs and "单说话人" in checkboxs):
        return "保留样本前置条件不满足", sample, audio

    annotation = {
        "clear": {
            "清晰": 1,
            "适中": 2,
            "模糊": 3,
        }[clear],
        "vocal_only": "纯人声" in checkboxs,
        "single_speaker": "单说话人" in checkboxs,
        "keep": "保留样本" in checkboxs,
    }
    update_sample(sample["_id"], annotation, user["_id"])

    sample = get_new_sample("audio-quality", user["_id"])
    if sample is None:
        return "所有样本已标注完毕", None, None

    return "已获取样本", sample, sample["data"]


def login_panel():
    with gr.Column(scale=3):
        username = gr.Textbox(label="用户名")
        password = gr.Textbox(label="密码", type="password")
        login = gr.Button(value="登录")
        status = gr.Label(label="状态", value="未登录")

    return username, password, login, status


def annotation_panel(username, password):
    with gr.Column(scale=9):
        audio = gr.Audio(label="音频", type="filepath")

        with gr.Row():
            with gr.Column(scale=2):
                clear = gr.Radio(["清晰", "适中", "模糊"], label="音频质量")

            with gr.Column(scale=4):
                checkboxs = gr.CheckboxGroup(
                    ["纯人声", "单说话人", "保留样本"], value=["保留样本"], label="音频类型"
                )

        submit = gr.Button(value="提交")

    return audio, clear, checkboxs, submit


def main():
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        current_sample = gr.State(None)

        with gr.Row():
            username, password, login, status = login_panel()
            audio, clear, checkboxs, submit = annotation_panel(username, password)

        login.click(
            call_login_user, [username, password], [status, current_sample, audio]
        )
        submit.click(
            call_submit_sample,
            [username, password, current_sample, audio, clear, checkboxs],
            [status, current_sample, audio],
        )

    app.launch(
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
