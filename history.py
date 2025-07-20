class Chat_history():
    def __init__(self, history_limit: int = 20) -> None:
        self.history = {}
        self.cur_answer_id = 0
        self.history_limit = history_limit

    def add_answer(self, pipeline_answer: dict) -> None:
        if len(self.history.keys()) == self.history_limit:
            oldest_id = min(self.history.keys())
            del self.history[oldest_id]
        self.history[self.cur_answer_id] = pipeline_answer
        self.cur_answer_id += 1

    def get_history(self) -> dict:
        return self.history

    def delete_history(self) -> None:
        self.history = {}
        self.cur_answer_id = 0


def make_paraphrase_prompt(
    query: str,
    chat_history: Chat_history
) -> str:
    history = chat_history.get_history()
    if not history:
        return f"Текущий вопрос пользователя:\n{query}"
    prompt = (
        "Текущий вопрос пользователя:\n"
        f"{query}\n"
        "Предыдущая история общения:\n"
    )
    answer_ids = sorted(history.keys())
    history_blocks = []
    for id in answer_ids:
        history_prompt = (
            "Вопрос пользователя:\n"
            f"{history[id]['query']}\n"
            "Ответ ассистента:\n"
            f"{history[id]['answer']}"
        )
        history_blocks.append(history_prompt)
    return prompt + "\n\n" + "\n\n".join(history_blocks)
