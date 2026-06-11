#!/usr/bin/env python3
"""Test async connection to local server"""
import asyncio
import sys
import os
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

async def test_async_v1():
    """Test with AsyncOpenAI"""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8765/v1",
        api_key="dummy_key_for_local_server",
        timeout=60,
    )
    
    model_name = "claude-opus-4-6"
    
    try:
        print(f"[AsyncOpenAI] 尝试连接到本地 server...")
        print(f"  Base URL: {client.base_url}")
        print(f"  Model: {model_name}")
        
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "你好，请回复'连接成功'。"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        print("\n✓ [AsyncOpenAI] 连接成功！")
        print(f"回复内容: {completion.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"\n✗ [AsyncOpenAI] 连接失败:")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_v2():
    """Test with extra_body parameters like in sampler.py"""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8765/v1",
        api_key="dummy_key_for_local_server",
        timeout=60,
    )
    
    model_name = "claude-opus-4-6"
    
    try:
        print(f"\n[AsyncOpenAI + extra_body] 尝试连接到本地 server...")
        
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "你好，请回复'连接成功'。"}
            ],
            max_tokens=50,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        
        print("\n✓ [AsyncOpenAI + extra_body] 连接成功！")
        print(f"回复内容: {completion.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"\n✗ [AsyncOpenAI + extra_body] 连接失败:")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multiple_concurrent():
    """Test multiple concurrent requests"""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8765/v1",
        api_key="dummy_key_for_local_server",
        timeout=60,
    )
    
    model_name = "claude-opus-4-6"
    
    async def single_request(i):
        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": f"请回复'请求{i}成功'。"}
                ],
                max_tokens=50,
            )
            return True, completion.choices[0].message.content
        except Exception as e:
            return False, str(e)
    
    try:
        print(f"\n[Concurrent Requests] 尝试5个并发请求...")
        
        tasks = [single_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for ok, _ in results if ok)
        print(f"\n✓ 完成: {success_count}/5 请求成功")
        
        for i, (ok, msg) in enumerate(results):
            status = "✓" if ok else "✗"
            print(f"  {status} 请求{i}: {msg[:50]}...")
        
        return success_count == 5
        
    except Exception as e:
        print(f"\n✗ [Concurrent Requests] 失败:")
        print(f"  错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("=" * 60)
    print("异步连接诊断测试")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic async connection
    results.append(("基础异步连接", await test_async_v1()))
    
    # Test 2: With extra_body parameters
    results.append(("带extra_body参数", await test_async_v2()))
    
    # Test 3: Multiple concurrent requests
    results.append(("并发请求", await test_multiple_concurrent()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
