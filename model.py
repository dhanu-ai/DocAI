# model.py
import dotenv
import google.generativeai as genai

# Configure the API key for the generative AI
gemini_api_key = dotenv.get_key(".env", "GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)

# Define the generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

def model(query, history):
    # Format history for Google Generative AI
    formatted_history = []
    for item in history:
        role = "user" if item["role"] == "user" else "model"
        formatted_history.append({
            "role": role,
            "parts": [{"text": item["content"]}]
        })

    # Create the model with the specified generation configuration
    generative_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    
    # Start or continue the chat session with history
    chat_session = generative_model.start_chat(history=formatted_history)

    # Prepare the message with system instruction and user query
    message = f"""
    System: You're a doctor. The individual will ask the details about the test performed.
    Your task is:
    - Name the possible disease according to the test in Bold Letters.
    - Give some home remedies if possible.
    - Provide a diet plan to avoid consuming those food items which may affect the disease negatively; the main goal of the diet plan is to minimize the risk of disease.
    - Give some advice which may help in preventing the disease.
    
    Remember:
        You're not a certified doctor. You're helping the user to find the possible disease and home remedy. Most importantly the user is seeking help before consulting with a doctor.
        In the end, highlight the importance of consulting a doctor according to the user's situation.
    
    User: {query}
    """

    # Get the response from the model
    response = chat_session.send_message(message)
    response_text = response.text

    return response_text
