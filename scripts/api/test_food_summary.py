
from food_summary import get_food_summary

# test the food summary function
def test_food_summary():
    """Simple test function."""
    print("\nFood Nutrition Info")
    print("Enter 'quit' to exit")
    
    while True:
        food = input("\nEnter a food: ").strip()
        if food.lower() == 'quit':
            break
            
        info = get_food_summary(food)
        if info:
            print(f"\n{info}")
        else:
            print("Failed to get nutrition info")

if __name__ == "__main__":
    test_food_summary() 