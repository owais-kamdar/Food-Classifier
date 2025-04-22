"""
This script is used to get the summary of a food item from OpenAI.
"""

import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_food_summary(food_name):
    """
    Get basic nutrition information for a food item.
    Args: food_name (str): Name of the food item
    Returns: str: Nutrition information from OpenAI
    """
    # Load API key
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Simple prompt
    prompt = f"""
    Food: {food_name}
    
    Please provide the following information in markdown format:
    
    #### Nutrition Facts
    - Serving Size: [just the number]
    - Calories: [just the number based on the serving size]
    - Protein: [just the number based on the serving size]
    - Carbs: [just the number based on the serving size]
    - Fat: [just the number based on the serving size]
    - Sodium: [just the number based on the serving size]
    - Sugar: [just the number based on the serving size]
    
    #### Origin & History
    [Brief description of the food's origin and history]
    
    #### Health Tip
    [Practical health advice about the food]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a nutrition expert providing simple, accurate food information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error getting nutrition info: {e}")
        return None 