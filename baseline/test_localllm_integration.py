#!/usr/bin/env python3
"""Test LocalLLM integration"""
import asyncio
import sys
import os
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent / "llm-srbench" / "methods"))

async def test_localllm_integration():
    """Test LocalLLM class directly"""
    from llmsr import config as config_lib
    from llmsr import sampler
    
    # Create config matching YAML
    cfg = config_lib.Config(
        use_api=True,
        api_model="claude-opus-4-6",
        samples_per_prompt=1,  # Test with 1 sample
    )
    
    # Create LocalLLM instance
    llm = sampler.LocalLLM(
        samples_per_prompt=1,
        local_llm_url="http://127.0.0.1:8765/v1/",
        api_url="http://127.0.0.1:8765/v1/",
        api_key="dummy_key",
    )
    
    print("[LocalLLM Integration Test]")
    print(f"  Config use_api: {cfg.use_api}")
    print(f"  Config api_model: {cfg.api_model}")
    print(f"  LLM client base_url: {llm.client.base_url}")
    
    # Test prompt
    test_prompt = """You are a helpful assistant.
Complete the 'equation' function below:

def equation(x, y):
    return"""
    
    try:
        print(f"\n[1/3] Testing async_draw_single_sample...")
        result = await llm.async_draw_single_sample(test_prompt, cfg)
        print(f"✓ Success! Response length: {len(result)} chars")
        print(f"  Response preview: {result[:100]}...")
        
        # Test multiple calls
        print(f"\n[2/3] Testing multiple sequential calls...")
        for i in range(3):
            result = await llm.async_draw_single_sample(test_prompt, cfg)
            print(f"  ✓ Call {i+1}: {len(result)} chars")
        
        # Test concurrent calls
        print(f"\n[3/3] Testing concurrent calls...")
        tasks = [
            llm.async_draw_single_sample(test_prompt, cfg)
            for _ in range(3)
        ]
        results = await asyncio.gather(*tasks)
        print(f"  ✓ All {len(results)} concurrent calls completed")
        
        print("\n✓ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed:")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("=" * 70)
    print("LocalLLM Integration Test")
    print("=" * 70)
    success = await test_localllm_integration()
    print("=" * 70)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
