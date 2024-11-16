from flask import Flask, request, jsonify
from main import handle_query

# instancia o servidor flask
app = Flask(__name__)

# lista para armazenar as respostas
chat_answers = []

# rota inicial
@app.route("/")
def init_server():
    return "Servidor sendo executado."

# rota para recuperar as respostas
@app.route("/chat-answers", methods=["GET"])
def return_answers():
    return jsonify(chat_answers), 200

# rota para fazer as perguntas
@app.route("/ask-chat", methods=["POST"])
def ask_chat():
    data = request.get_json()
    pergunta = data.get("pergunta") # pergunta feita pelo usuário

    if not pergunta:
        return jsonify({"error": "Pergunta não fornecida"}), 400

    resposta = handle_query(pergunta) # executa a função handle_query passando a pergunta como parâmetro para obter a resposta

    chat_answers.append({
        "pergunta": pergunta,
        "resposta": resposta
    }) # formata o json

    return jsonify({"resposta": resposta}), 200

# executa o servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0")