import streamlit as st
import os
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from groq import Groq

# Set up the Groq API Key (replace with your actual API key)
api_key = 'Your_Groq_API'  # Replace with your Groq API key

# Initialize the Groq client
client = Groq(api_key=api_key)

# Define image preprocessing and ResNet-50 model
model = models.resnet50(pretrained=True)
model.eval()

# Define image transformations for preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify the image using ResNet-50
def classify_image(image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Function to get class names from ImageNet (for image classification)
def get_class_names():
    url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    return response.json()

# Function to interact with Groq chatbot API
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant. Only respond to questions about animals. If the input is not related to animals, politely decline to answer."}
]

def chat_with_groq(user_message, classification_result=None):
    global conversation_history  # Use the global conversation history
    
    # If classification result is provided, include it in the prompt
    if classification_result:
        user_message = f"{user_message} about {classification_result}"  # Append classified object to the user message
    
    # Append the new user message to the conversation history
    conversation_history.append({"role": "user", "content": user_message})

    # Now send the entire conversation history to Groq chatbot
    try:
        chat_completion = client.chat.completions.create(
            messages=conversation_history,
            model="llama3-8b-8192",  # Use the Groq model
        )
        # Get the chatbot's response
        chatbot_response = chat_completion.choices[0].message.content
        
        # Append the chatbot's response to the conversation history
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        
        return chatbot_response

    except Exception as e:
        return f"Oops! Something went wrong. Error: {str(e)}"

# Streamlit Interface
def main():
    st.title("Image Classification🐼 with Chatbot🤖")

    # Image upload button during chat
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify the image
        class_id = classify_image(image)
        class_names = get_class_names()

        if class_id < len(class_names):
            classified_as = class_names[class_id]
            st.write(f"This is an image of {classified_as}. You can ask me if you want to know more about {classified_as}.")
        else:
            classified_as = "Unknown"
            st.write("Classification result is unknown")
        
        # Set session state to indicate image is uploaded
        st.session_state.image_uploaded = True

    # Show the input field only after image is uploaded
    if 'image_uploaded' in st.session_state and st.session_state.image_uploaded:
        # Creating a placeholder for the bot's response, this will be updated dynamically
        response_placeholder = st.empty()

        # Initialize session state for user message if not already set
        if 'user_message' not in st.session_state:
            st.session_state.user_message = ""  # Initialize the session state for user message

        # Input field for chatbot interaction
        user_message = st.text_input("You: ", value=st.session_state.user_message, key="user_message_input")

        # Only process the message after the user submits it
        if user_message:
            # Send the user's message to the chatbot along with the classification result
            bot_response = chat_with_groq(user_message, classification_result=classified_as)

            # Display the bot's response in the placeholder with scroll and max height
            response_placeholder.markdown(f'''
                <div style="text-align: left; color: #333333; background-color: #e0e0e0; border-radius: 10px; padding: 10px; margin: 5px; height: 150px; max-height: 150px; overflow-y: scroll;">
                    {bot_response}
                </div>
            ''', unsafe_allow_html=True)

            # After submitting, clear the input text
            st.session_state.user_message = ""  # Clear the text input after submission

# Custom CSS for styling chat messages
st.markdown("""
    <style>
        .streamlit-expanderHeader {
            font-size: 1.5em;
            font-weight: bold;
        }
        .chat-box {
            border: 1px solid #ddd;
            padding: 10px;
            font-size: 18px;
            font-families: Serif;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
