"""
Simple Chatbot
Basic conversational chatbot using rule-based approach
"""

import random

# simple responses dictionary
responses = {
    'hello': ['Hi there!', 'Hello!', 'Hey!'],
    'how are you': ['I am doing well!', 'Great, thanks!', 'Fine, how about you?'],
    'bye': ['Goodbye!', 'See you later!', 'Bye!'],
    'default': ['I did not understand that.', 'Can you rephrase?', 'Tell me more.']
}

def get_response(user_input):
    """Get chatbot response"""
    user_input = user_input.lower()
    
    for key in responses:
        if key in user_input:
            return random.choice(responses[key])
    
    return random.choice(responses['default'])

# chat loop
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'quit':
#         break
#     response = get_response(user_input)
#     print(f"Bot: {response}")

print("Simple chatbot ready. Run the chat loop to interact.")
