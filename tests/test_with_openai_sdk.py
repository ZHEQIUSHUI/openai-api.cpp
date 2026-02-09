#!/usr/bin/env python3
"""
Complete test suite for OpenAI API compatible server using Python SDK.

Usage:
    # First start the server:
    cd build && ./openai_server 18099
    
    # Then run this test:
    python3 tests/test_with_openai_sdk.py
"""

import sys
import time
from openai import OpenAI, BadRequestError

BASE_URL = "http://localhost:18099"
API_KEY = "mock-api-key"


def test_models_list():
    """Test: GET /models"""
    print("\n[1/8] Testing Models List...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    
    assert "gpt-4" in model_ids, "gpt-4 not found"
    assert "text-embedding-ada-002" in model_ids, "text-embedding-ada-002 not found"
    print(f"    ✓ Found {len(model_ids)} models: {model_ids[:5]}...")
    return True


def test_chat_completion():
    """Test: POST /chat/completions (non-streaming)"""
    print("\n[2/8] Testing Chat Completion (Non-streaming)...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    assert response.choices[0].message.role == "assistant"
    assert len(response.choices[0].message.content) > 0
    print(f"    ✓ Response: {response.choices[0].message.content[:60]}...")
    return True


def test_chat_completion_streaming():
    """Test: POST /chat/completions (streaming)"""
    print("\n[3/8] Testing Chat Completion (Streaming)...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True
    )
    
    collected = []
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        if chunk.choices[0].delta.content:
            collected.append(chunk.choices[0].delta.content)
    
    full_text = "".join(collected)
    assert len(full_text) > 0, "No content received"
    print(f"    ✓ Received {chunk_count} chunks, text: {full_text[:60]}...")
    return True


def test_embeddings():
    """Test: POST /embeddings"""
    print("\n[4/8] Testing Embeddings...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="Hello world"
    )
    
    dims = len(response.data[0].embedding)
    assert dims > 0, "Empty embedding"
    print(f"    ✓ Embedding dimensions: {dims}")
    print(f"    ✓ First 5 values: {response.data[0].embedding[:5]}")
    return True


def test_embeddings_batch():
    """Test: POST /embeddings (batch)"""
    print("\n[5/8] Testing Embeddings (Batch)...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=["Hello", "World", "Test"]
    )
    
    assert len(response.data) == 3, f"Expected 3 embeddings, got {len(response.data)}"
    print(f"    ✓ Got {len(response.data)} embeddings")
    return True


def test_tts():
    """Test: POST /audio/speech"""
    print("\n[6/8] Testing Text-to-Speech...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="Hello, this is a test."
    )
    
    audio_size = len(response.content)
    assert audio_size > 0, "No audio data"
    print(f"    ✓ Received {audio_size} bytes of audio")
    
    # Save to file
    output_path = "/tmp/test_tts_output.mp3"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"    ✓ Saved to {output_path}")
    return True


def test_image_generation():
    """Test: POST /images/generations"""
    print("\n[7/8] Testing Image Generation...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    response = client.images.generate(
        model="dall-e-3",
        prompt="A cute cat sitting on a table",
        n=1,
        size="1024x1024"
    )
    
    assert response.data[0].url is not None
    print(f"    ✓ Image URL: {response.data[0].url}")
    return True


def test_error_handling():
    """Test: Error handling"""
    print("\n[8/8] Testing Error Handling...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    try:
        # Missing 'messages' field
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[]  # Empty messages
        )
        print("    ✗ Should have raised an error")
        return False
    except Exception as e:
        print(f"    ✓ Got expected error: {type(e).__name__}")
        return True


def main():
    print("=" * 60)
    print("OpenAI API Compatible Server - Python SDK Test Suite")
    print("=" * 60)
    print(f"Server URL: {BASE_URL}")
    
    tests = [
        ("Models List", test_models_list),
        ("Chat Completion", test_chat_completion),
        ("Chat Streaming", test_chat_completion_streaming),
        ("Embeddings", test_embeddings),
        ("Embeddings Batch", test_embeddings_batch),
        ("Text-to-Speech", test_tts),
        ("Image Generation", test_image_generation),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    print("=" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
