import json
import typing as t

from langchain_core.pydantic_v1 import BaseModel


TBaseModel = t.TypeVar("TBaseModel", bound=BaseModel)


JSON_FORMAT_INSTRUCTIONS_RU = """Вывод должен быть «правильно форматированным» экземпляром JSON, который соответствует схеме JSON ниже.

Например, для схемы {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
объект {{"foo": ["bar", "baz"]}} является «правильно форматированным». Объект {{"properties": {{"foo": ["bar", "baz"]}}}} НЕ является «правильно форматированным».

Вот пример выходящей схемы JSON:
```
{schema}
```

Не возвращай никаких преамбул или объяснений, верните только чистую строку JSON, окруженную тройными обратными кавычками (```)."""


def get_json_format_instructions_ru(pydantic_object: t.Type[TBaseModel]) -> str:
    # Copy schema to avoid altering original Pydantic schema.
    schema = {k: v for k, v in pydantic_object.schema().items()}

    # Remove extraneous fields.
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    # Ensure json in context is well-formed with double quotes.
    schema_str = json.dumps(reduced_schema)

    resp = JSON_FORMAT_INSTRUCTIONS_RU.format(schema=schema_str)
    return resp
