# model.py
import dotenv
import google.generativeai as genai

# Configure the API key for the generative AI
gemini_api_key = dotenv.get_key(".env", "GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)

# Define the generation configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

def model(info, history):
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
    System: You're task is to refine and fact check the {info} and make is sound like a professinal doctor advise. Use the give information as the main source no need to provide any additional information.
    Remember:
        -The information is based on medical report so do not make any additional information or make up things.
        -The information is provided is not from real doctors.
    User: {info}
    """

    # Get the response from the model
    response = chat_session.send_message(message)
    response_text = response.text

    return response_text
