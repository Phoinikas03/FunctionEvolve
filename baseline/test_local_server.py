from openai import OpenAI
import os

def test_local_server_v1():
    # 初始化客户端，指向本地 server
    client = OpenAI(
        base_url="http://127.0.0.1:8765/v1",
        api_key="dummy_key_for_local_server" # 本地通常不需要真实key
    )

    model_name = "claude-opus-4-6"

    try:
        print(f"尝试连接到本地 server ({client.base_url})，使用模型: {model_name}...")
        
        # 发送测试请求
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "你好，请回复'连接成功'。"}
            ],
            max_tokens=50,
            temperature=0.7
        )

        print("\n--- 连接成功！---")
        print("模型回复内容:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\n[错误] 连接或请求失败:")
        print(e)

if __name__ == "__main__":
    test_local_server_v1()