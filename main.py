import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go
import pandas as pd
import json
import os
import time
from collections import defaultdict

@st.cache_resource
def load_model(model_name):
    """Load the pre-trained model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length, temperature, top_k, top_p, repetition_penalty):
    """Generate text using the LLM model and track performance metrics."""
    try:
        start_time = time.time()
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        end_time = time.time()
        
        # Calculate metrics
        generation_time = end_time - start_time
        input_token_count = len(input_ids[0])
        output_token_count = len(output[0])
        
        return generated_text, generation_time, input_token_count, output_token_count
    except Exception as e:
        st.error(f"An error occurred during text generation: {str(e)}")
        return None, None, None, None

def plot_token_probabilities(model, tokenizer, output_text):
    """Create a bar chart of token probabilities."""
    tokens = tokenizer.encode(output_text)
    token_probs = torch.softmax(model(torch.tensor([tokens])).logits[0, -1, :], dim=-1)
    top_k = 10
    top_probs, top_indices = torch.topk(token_probs, k=top_k)
    
    df = pd.DataFrame({
        'Token': [tokenizer.decode([idx.item()]) for idx in top_indices],
        'Probability': top_probs.detach().numpy()
    })
    
    fig = go.Figure(data=[go.Bar(x=df['Token'], y=df['Probability'])])
    fig.update_layout(title='Top 10 Token Probabilities', xaxis_title='Token', yaxis_title='Probability')
    return fig

def save_conversation(conversation, filename):
    """Save the conversation history to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(conversation, f)

def load_conversation(filename):
    """Load the conversation history from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

# Initialize session state for metrics
if 'metrics' not in st.session_state:
    st.session_state.metrics = defaultdict(list)
    st.session_state.metrics['model_usage'] = defaultdict(int)

# Set up the Streamlit interface
st.title("Interactive LLM-powered Text Generation")
st.markdown("Enter a prompt and adjust the parameters to generate text using a pre-trained language model.")

# Model selection
model_options = {
    "GPT-2": "gpt2",
    "GPT-2 Medium": "gpt2-medium"
}
selected_model = st.selectbox("Select a model:", list(model_options.keys()))

# Load the selected model and tokenizer
model, tokenizer = load_model(model_options[selected_model])

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_prompt = st.text_area("Enter your prompt:", "Once upon a time")

# Parameter adjustment
st.subheader("Advanced Parameter Tuning")
col1, col2 = st.columns(2)
with col1:
    max_length = st.slider("Maximum response length:", min_value=50, max_value=500, value=200, step=50)
    temperature = st.slider("Temperature (creativity):", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    top_k = st.slider("Top-k sampling:", min_value=1, max_value=100, value=50, step=1)
with col2:
    top_p = st.slider("Top-p (nucleus) sampling:", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
    repetition_penalty = st.slider("Repetition penalty:", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

# Generate button
if st.button("Generate Text") or not st.session_state.metrics['generation_time']:
    with st.spinner("Generating text..."):
        generated_text, generation_time, input_token_count, output_token_count = generate_text(
            model, tokenizer, user_prompt, max_length, temperature, top_k, top_p, repetition_penalty
        )
    
    if generated_text:
        st.subheader("Generated Text:")
        st.write(generated_text)
        
        # Add to conversation history
        st.session_state.conversation_history.append({"prompt": user_prompt, "response": generated_text})
        
        # Update metrics
        st.session_state.metrics['generation_time'].append(generation_time)
        st.session_state.metrics['input_token_count'].append(input_token_count)
        st.session_state.metrics['output_token_count'].append(output_token_count)
        st.session_state.metrics['model_usage'][selected_model] += 1
        
        # Display token probabilities
        st.subheader("Token Probability Distribution:")
        fig = plot_token_probabilities(model, tokenizer, generated_text)
        st.plotly_chart(fig)
    else:
        st.warning("Failed to generate text. Please try again.")

# Display conversation history
st.subheader("Conversation History")
for i, entry in enumerate(st.session_state.conversation_history):
    st.text(f"Prompt {i+1}: {entry['prompt']}")
    st.text(f"Response {i+1}: {entry['response']}")
    st.markdown("---")

# Save conversation button
if st.button("Save Conversation"):
    save_conversation(st.session_state.conversation_history, "conversation_history.json")
    st.success("Conversation saved successfully!")

# Load conversation button
if st.button("Load Conversation"):
    loaded_conversation = load_conversation("conversation_history.json")
    if loaded_conversation:
        st.session_state.conversation_history = loaded_conversation
        st.success("Conversation loaded successfully!")
    else:
        st.warning("No saved conversation found.")

# Analytics Dashboard
st.subheader("Analytics Dashboard")
col1, col2, col3 = st.columns(3)

if st.session_state.metrics['generation_time']:
    with col1:
        st.metric("Average Generation Time", f"{sum(st.session_state.metrics['generation_time']) / len(st.session_state.metrics['generation_time']):.2f} seconds")

    with col2:
        st.metric("Average Input Token Count", f"{sum(st.session_state.metrics['input_token_count']) / len(st.session_state.metrics['input_token_count']):.0f} tokens")

    with col3:
        st.metric("Average Output Token Count", f"{sum(st.session_state.metrics['output_token_count']) / len(st.session_state.metrics['output_token_count']):.0f} tokens")

    # Model usage chart
    model_usage = pd.DataFrame.from_dict(dict(st.session_state.metrics['model_usage']), orient='index', columns=['Usage'])
    model_usage.index.name = 'Model'
    model_usage.reset_index(inplace=True)

    fig = go.Figure(data=[go.Pie(labels=model_usage['Model'], values=model_usage['Usage'], hole=.3)])
    fig.update_layout(title='Model Usage Distribution')
    st.plotly_chart(fig)

    # Performance over time
    if len(st.session_state.metrics['generation_time']) > 1:
        performance_df = pd.DataFrame({
            'Generation': range(1, len(st.session_state.metrics['generation_time']) + 1),
            'Generation Time': st.session_state.metrics['generation_time'],
            'Input Token Count': st.session_state.metrics['input_token_count'],
            'Output Token Count': st.session_state.metrics['output_token_count']
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=performance_df['Generation'], y=performance_df['Generation Time'], mode='lines+markers', name='Generation Time'))
        fig.add_trace(go.Scatter(x=performance_df['Generation'], y=performance_df['Input Token Count'], mode='lines+markers', name='Input Token Count'))
        fig.add_trace(go.Scatter(x=performance_df['Generation'], y=performance_df['Output Token Count'], mode='lines+markers', name='Output Token Count'))
        fig.update_layout(title='Performance Metrics Over Time', xaxis_title='Generation', yaxis_title='Value')
        st.plotly_chart(fig)
else:
    st.info("Generate some text to see analytics!")

# Error handling and user feedback
st.markdown("---")
st.markdown("### Tips:")
st.markdown("- If the generated text is cut off, try increasing the maximum response length.")
st.markdown("- Adjust the temperature to control the randomness of the output. Higher values lead to more creative but potentially less coherent text.")
st.markdown("- Top-k sampling limits the number of possible next tokens to consider.")
st.markdown("- Top-p sampling limits the cumulative probability of next tokens to consider.")
st.markdown("- Repetition penalty discourages the model from repeating the same phrases.")
st.markdown("- If you encounter any errors, try refreshing the page or entering a different prompt.")

# Add some information about the models
st.sidebar.title("About")
st.sidebar.info(f"This application uses the {selected_model} model from Hugging Face Transformers to generate text based on your input. The model has been trained on a diverse range of internet text and can generate coherent continuations of prompts.")
st.sidebar.warning("Note: The generated text is produced by an AI model and may contain biases or inaccuracies. Use the output responsibly and critically.")
