import os
import asyncio
import sys
from openai import AsyncOpenAI

# Настройка кодировки для Windows, чтобы не было UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Настройка OpenAI-совместимого клиента для RouterAiru
# Ожидаются переменные окружения:
# ROUTERAIRU_API_KEY - ваш API ключ
# ROUTERAI_API_ENDPOINT - базовый URL API (например, https://api.router.ai.ru/v1)
openai_api_key = os.getenv('ROUTERAIRU_API_KEY')
openai_api_base = os.getenv('ROUTERAI_API_ENDPOINT')

# Инициализируем асинхронный клиент
client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)

async def send_request(messages, model="openai/gpt-4o-mini", **kwargs):
    """
    Отправляет запрос к API и возвращает объект ответа.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 1.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            stop=kwargs.get("stop", None),
            stream=kwargs.get("stream", False),
        )
        return response
    except Exception as e:
        print(f"Ошибка при обращении к API: {e}")
        return None

async def main():
    # Проверка настроек
    if not openai_api_key:
        print("Внимание: ROUTERAIRU_API_KEY не установлен в переменных окружения.")
    if not openai_api_base:
        print("Внимание: ROUTERAI_API_ENDPOINT не установлен, будет использован адрес по умолчанию.")

    # Пример сообщений (используем тот же вопрос, что и в других тестах)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Кто лучший французский художник? Напиши кратко."}
    ]

    print(f"Отправка запроса к RouterAiru (Base URL: {openai_api_base or 'default'})...")
    
    # Вызываем функцию запроса
    response = await send_request(messages)

    if response:
        print("\n--- Ответ от модели ---")
        print(response.choices[0].message.content)
        print("-----------------------")
    else:
        print("\nНе удалось получить ответ. Проверьте настройки подключения.")

if __name__ == "__main__":
    # Запускаем асинхронный цикл
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
