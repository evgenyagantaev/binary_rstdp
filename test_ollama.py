import ollama

# Убедись, что само приложение Ollama запущено в фоне!

# Простой запрос (чат)
response = ollama.chat(model='gemma3:1b', messages=[
  {
    'role': 'user',
    'content': 'write a command of light turning off in the kitchen in structured format json',
  },
])

# Вывод ответа
print(response['message']['content'])