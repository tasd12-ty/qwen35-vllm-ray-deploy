import time
from openai import OpenAI

HEAD_IP = "10.0.0.1"  # change to your head node IP

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://{HEAD_IP}:8000/v1",
    timeout=3600,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                },
            },
            {
                "type": "text",
                "text": "Describe the image and list the main objects.",
            },
        ],
    }
]

start = time.time()
resp = client.chat.completions.create(
    model="qwen3.5-mm-1m",
    messages=messages,
    max_tokens=512,
)
print(f"Latency: {time.time() - start:.2f}s")
print(resp.choices[0].message.content)
