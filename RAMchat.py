
import streamlit as st
from retriever import respond
from langchain_core.messages import HumanMessage, AIMessage

# Initialisation de la conversation
WELCOME_MESSAGE = (
    "Bonjour, je suis l'assistant virtuel de la RAM. "
    "Comment puis-je vous aider aujourd'hui?"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]


# Construction de l'historique pour LangChain
def construire_prompt_session(messages, max_messages=10):

    recent_messages = messages[-max_messages:]

    prompt_messages = []

    for msg in recent_messages:
        if msg["role"] == "user":
            prompt_messages.append(
                HumanMessage(content=msg["content"])
            )
        elif msg["role"] == "assistant":
            prompt_messages.append(
                AIMessage(content=msg["content"])
            )

    return prompt_messages


# Génération de la réponse
def generer_response(messages):
    return respond(messages)


# Interface
st.title("Assistant Virtuel de la RAM")

# Afficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Saisie utilisateur
if prompt := st.chat_input("Posez votre question ici..."):

    # Ajouter la question
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    # Préparer l'historique
    prompt_messages = construire_prompt_session(
        st.session_state.messages
    )

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Je prépare votre réponse..."):
            try:
                response_content = generer_response(
                    prompt_messages
                )
            except Exception as e:
                response_content = f"Une erreur est survenue : {e}"

        st.write(response_content)

    # Sauvegarder la réponse
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content
    })


# Réinitialiser la conversation
if st.button("Effacer la conversation"):
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]
    st.rerun()
