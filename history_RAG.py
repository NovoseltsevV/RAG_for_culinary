from RAGModel import load_system_prompt, RAGmodel
from history import Chat_history, make_paraphrase_prompt


class RAG_with_history():
    def __init__(
        self,
        rag_model: RAGmodel,
        paraphrase_model,
        history_limit: int = 20,
        paraphrase_model_cfg: dict = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
            }
    ) -> None:
        self.rag_model = rag_model
        self.paraphrase_model = paraphrase_model
        self.chat_history = Chat_history(history_limit)
        self.paraphrase_model_cfg = paraphrase_model_cfg
        self.paraphrase_prompt = load_system_prompt(
            filepath="paraphrase_prompt.txt"
        )
        self.paraphrase_history = {}
        self.cur_query_id = 0

    def start_conversation(self):
        self.chat_history.delete_history()
        self.paraphrase_history = {}
        self.cur_query_id = 0
        print(
            (
                "Начат новый разговор с ассистентом. "
                "Предыдущая история запросов удалена"
            )
        )

    def ask_question(self, query: str) -> str:
        if self.chat_history.get_history():
            parphrase_query = make_paraphrase_prompt(query, self.chat_history)
            parphrase_messages = [
                {"role": "system", "content": self.paraphrase_prompt},
                {"role": "user", "content": parphrase_query}
            ]
            query = self.paraphrase_model(
                parphrase_messages,
                **self.paraphrase_model_cfg
            )[0]['generated_text'][-1]['content']
            self.paraphrase_history[self.cur_query_id] = query
        answer = self.rag_model.generate_recipe(query)
        pipeline_answer = {
            "query": query,
            "answer": answer
        }
        self.chat_history.add_answer(pipeline_answer)
        self.cur_query_id += 1
        return answer
