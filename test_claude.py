from decouple import config
from groq import Groq

client = Groq(
    api_key=config("GROQ_API_KEY")
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Give me 5 names of animals"
        }
    ],
    model="openai/gpt-oss-20b",
)

print(chat_completion.choices[0].message.content)