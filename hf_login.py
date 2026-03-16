"""HuggingFace Login via Python"""
from huggingface_hub import login
import os

print("="*70)
print("HuggingFace Login")
print("="*70)
print("\nGet your token from: https://huggingface.co/settings/tokens")
print("Create a READ token (write access not needed)")
print("\nPaste your token below (it will be saved in ~/.huggingface/token):")
print("-"*70)

token = input("Token: ").strip()

if token:
    try:
        login(token=token, add_to_git_credential=False)
        print("\n✅ Successfully logged in to HuggingFace!")
        print("   Token saved to: ~/.huggingface/token")
        print("\n   You can now download models without rate limits.")
    except Exception as e:
        print(f"\n❌ Login failed: {e}")
else:
    print("\n❌ No token provided. Skipping login.")
    print("   You can still use models but with rate limits.")

print("\n" + "="*70)