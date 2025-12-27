from mistralai import Mistral
import os


with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as mistral:

    res = mistral.chat.complete(model="mistral-large-latest", messages=[
        {
            "content": "Who is the best French painter?",
            "role": "user",
        },
    ], stream=False)

    # Handle response
    print(res.choices[0].message.content)

