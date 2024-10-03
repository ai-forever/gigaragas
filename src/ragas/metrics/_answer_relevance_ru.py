from ragas.llms.prompt import Prompt
from ragas.llms.output_parser_ru import get_json_format_instructions_ru

from ragas.metrics._answer_relevance import ResponseRelevanceClassification, AnswerRelevancy

_output_instructions_answer_relevance_ru = get_json_format_instructions_ru(
    pydantic_object=ResponseRelevanceClassification
)

QUESTION_GEN_RU = Prompt(
    name="question_generation_ru",
    instruction="""Сгенерируй вопрос для данного "answer" и определи, является ли ответ уклончивым (noncommittal). Дай уклончивый (noncommittal) как 1, если ответ является уклончивым (noncommittal) и 0, если ответ является прямым (comittal). Уклончивый (noncommittal) ответ - это неясный, нечёткий или неоднозначный ответ. Например, «я не знаю», «я не могу» или «я не уверен» - это уклончивые (noncommittal) ответы.""",
    output_format_instruction=_output_instructions_answer_relevance_ru,
    examples=[
        {
            "answer": """Альберт Эйнштейн родился в Германии.""",
            "context": """Альберт Эйнштейн был теоретическим физиком, родившимся в Германии, который широко признан одним из величайших и самых влиятельных ученых всех времен.""",
            "output": ResponseRelevanceClassification.parse_obj(
                {
                    "question": "Где родился Альберт Эйнштейн?",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """Они могут изменять цвет кожи в зависимости от температуры окружающей среды.""",
            "context": """Недавнее научное исследование обнаружило новый вид лягушек в тропическом лесу Амазонки, который обладает уникальной способностью изменять цвет кожи в зависимости от температуры окружающей среды.""",
            "output": ResponseRelevanceClassification.parse_obj(
                {
                    "question": "Какими уникальными способностями обладает недавно обнаруженный вид лягушек?",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """Эверест""",
            "context": """Самая высокая гора на земле, измеренная от уровня моря, представляет собой известную вершину, расположенную в Гималаях.""",
            "output": ResponseRelevanceClassification.parse_obj(
                {
                    "question": "Какая самая высокая гора на Земле?",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """Я не знаю об инновациях в смартфоне, изобретенном в 2023 году, так как у меня нет информации после 2022 года.""",
            "context": """В 2023 году было объявлено о революционном изобретении: смартфон со временем батареи в течение одного месяца. Это в корне меняет способ, которым люди используют мобильные технологии.""",
            "output": ResponseRelevanceClassification.parse_obj(
                {
                    "question": "Какова была инновация в смартфоне, изобретенном в 2023 году?",
                    "noncommittal": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["answer", "context"],
    output_key="output",
    output_type="json",
)

answer_relevancy_ru = AnswerRelevancy()
answer_relevancy_ru.question_generation = QUESTION_GEN_RU
answer_relevancy_ru.strictness = 3
