import ast
from vector_db import LSHdatabase
from db_creation import semantic_search


def load_system_prompt(filepath: str = "system_prompt.txt") -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def make_prompt(
    query: str,
    relevant_recipes: list,
    sim_threshold: float
) -> str:
    recipe_prompts = []

    for recipe in relevant_recipes:
        similarity = recipe[1]
        data = recipe[2]

        if similarity < sim_threshold:
            recipe_prompt = "ТЫ НЕ ДОЛЖЕН ИСПОЛЬЗОВАТЬ ЭТОТ ПРИМЕР"
        else:
            ingredients = ", ".join(ast.literal_eval(data['ingredients']))
            recipe_prompt = (
                f"Сходство рецепта с запросом: {similarity:.3f}. "
                "Чем выше это значение, тем релевантнее подсказка.\n"
                f"Название блюда: {data['name']}.\n"
                f"Ингредиенты: {ingredients}.\n"
                f"Часть текста рецепта: {data['text']}."
            )

        recipe_prompts.append(recipe_prompt)

    prompt = (
        "Вопрос:\n"
        f"{query}\n\n"
        "Рецепты из базы данных, которые могут быть релевантны "
        "для ответа на вопрос:\n"
    )

    return prompt + "\n\n".join(recipe_prompts)


class RAGmodel():
    def __init__(
        self,
        pipeline,
        embedding_model,
        db: LSHdatabase
    ) -> None:
        self.model = pipeline
        self.embedding_model = embedding_model
        self.db = db
        self.system_prompt = load_system_prompt()

    def generate_recipe(
        self,
        query: str,
        rag_limit: int = 5,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        sim_threshold: float = 0.85
    ) -> str:
        relevant_recipes = semantic_search(
            self.db, query, self.embedding_model, limit=rag_limit
        )
        user_prompt = make_prompt(query, relevant_recipes, sim_threshold)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        answer = self.model(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        return answer[0]['generated_text'][-1]['content']
