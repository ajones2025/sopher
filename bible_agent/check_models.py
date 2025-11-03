# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:19:27 2025

@author: Administrator
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
else:
    genai.configure(api_key=api_key)

    # --- List Available Models ---
    print("Finding available models for your API key...")
    print("-" * 20)

    found_models = False
    for m in genai.list_models():
      # Check if the model supports the 'generateContent' method
      if 'generateContent' in m.supported_generation_methods:
        print(m.name)
        found_models = True

    if not found_models:
        print("No models supporting 'generateContent' were found for this API key.")

    print("-" * 20)