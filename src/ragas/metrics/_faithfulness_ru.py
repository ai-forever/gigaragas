from ragas.llms.prompt import Prompt
from ragas.llms.output_parser_ru import get_json_format_instructions_ru

from ragas.metrics.base import get_segmenter
from ragas.metrics._faithfulness import StatementsAnswers, StatementFaithfulnessAnswers, Faithfulness


_statements_output_instructions_ru = get_json_format_instructions_ru(StatementsAnswers)

LONG_FORM_ANSWER_PROMPT_RU = Prompt(
    name="long_form_answer_ru",
    output_format_instruction=_statements_output_instructions_ru,
    instruction="Учитывая \"question\", \"answer\" и \"sentences\" из \"answer\" проанализируй сложность каждого \"sentence\", указанного в \"sentences\", и разбей каждое sentence на одно или более понятных утверждения, в то же время не используй местоимений в каждом из утверждений. Форматируй результат в JSON.",
    examples=[
        {
            "question": "Кем был Альберт Эйнштейн и чем он наиболее известен?",
            "answer": "Он был физиком-теоретиком Германии, который широко признан одним из величайших и влиятельных физиков всех времен. Он был наиболее известен развитием теории относительности, он также внес важный вклад в разработку теории квантовой механики.",
            "sentences": """
0:Он был физиком-теоретиком Германии, который широко признан одним из величайших и влиятельных физиков всех времен. 
1:Он был наиболее известен развитием теории относительности, он также внес важный вклад в разработку теории квантовой механики.
        """,
            "analysis": StatementsAnswers.parse_obj(
                [
                    {
                        "sentence_index": 0,
                        "simpler_statements": [
                            "Альберт Эйнштейн был теоретическим физиком Германии.",
                            "Альберт Эйнштейн признан одним из величайших и самых влиятельных физиков всех времен.",
                        ],
                    },
                    {
                        "sentence_index": 1,
                        "simpler_statements": [
                            "Альберт Эйнштейн был наиболее известен за развитие теории относительности.",
                            "Альберт Эйнштейн также внес важный вклад в разработку теории квантовой механики.",
                        ],
                    },
                ]
            ).dicts(),
        }
    ],
    input_keys=["question", "answer", "sentences"],
    output_key="analysis",
    language="russian",
)

_faithfulness_output_instructions_ru = get_json_format_instructions_ru(
    StatementFaithfulnessAnswers
)


NLI_STATEMENTS_MESSAGE_RU = Prompt(
    name="nli_statements_ru",
    instruction="Твоя задача - судить о верности набора «утверждений», основанных на данном «контексте». Для каждого «утверждения» ты должен вынести вердикт: 1 - если «утверждение» непосредственно следует напрямую из контекста, или 0 - если «утверждение» непосредственно не следует напрямую из контекста.",
    output_format_instruction=_faithfulness_output_instructions_ru,
    examples=[
        {
            "context": """Джон - студент МФТИ. Он получает степень в области компьютерных наук. В этом семестре он зачислен в несколько курсов, включая структуры данных, алгоритмы и управление базами данных. Джон - усердный студент и проводит значительное количество времени, изучая и выполняя задания. Он часто остается поздно в библиотеке, чтобы работать над своими проектами.""",
            "statements": [
                "Джон специализируется по биологии.",
                "Джон проходит курс по искусственному интеллекту.",
                "Джон - усердный студент.",
                "У Джона есть работа на полставки.",
            ],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "Джон специализируется по биологии.",
                        "reason": "Основная специальность Джона явно упоминается как компьютерные науки. Нет информации, предполагающей, что он специализируется на биологии.",
                        "verdict": 0,
                    },
                    {
                        "statement": "Джон проходит курс по искусственному интеллекту.",
                        "reason": "В контексте упоминаются курсы, в которые в настоящее время зачисляется Джон, а искусственный интеллект не упоминается. Следовательно, нельзя сделать вывод, что Джон проходит курс по ИИ.",
                        "verdict": 0,
                    },
                    {
                        "statement": "Джон - усердный студент.",
                        "reason": "Контекст утверждает, что он тратит значительное количество времени на изучение и выполнение заданий. Кроме того, в нем упоминается, что он часто остается до поздна в библиотеке, чтобы работать над своими проектами, что подразумевает усердие.",
                        "verdict": 1,
                    },
                    {
                        "statement": "У Джона есть работа на полставки.",
                        "reason": "В контексте нет информации о том, что у Джона есть работа на полставки.",
                        "verdict": 0,
                    },
                ]
            ).dicts(),
        },
        {
            "context": """Фотосинтез - это процесс, используемый растениями, водорослями и определенными бактериями для превращения световой энергии в химическую энергию.""",
            "statements": ["Альберт Эйнштейн был гением."],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "Альберт Эйнштейн был гением.",
                        "reason": "Контекст и утверждение никак не связаны.",
                        "verdict": 0,
                    }
                ]
            ).dicts(),
        },
    ],
    input_keys=["context", "statements"],
    output_key="answer",
    output_type="json",
    language="russian",
)

faithfulness_ru = Faithfulness()

faithfulness_ru.statement_prompt = LONG_FORM_ANSWER_PROMPT_RU
faithfulness_ru.nli_statements_message = NLI_STATEMENTS_MESSAGE_RU
faithfulness_ru.sentence_segmenter = get_segmenter(language="russian", clean=False)
