from flask import Flask, request, jsonify
from main import handle_query
from flask_cors import CORS

# instancia o servidor flask
app = Flask(__name__)

# Habilita CORS
CORS(app)

# rota inicial


@app.route("/")
def init_server():
    return "Servidor sendo executado."

# rota para fazer as perguntas


@app.route("/ask-chat", methods=["POST"])
def ask_chat():
    data = request.get_json()
    pergunta = data.get("pergunta")  # pergunta feita pelo usuário

    if not pergunta:
        return jsonify({"error": "Pergunta não fornecida"}), 400

    # Chama a função handle_query para processar a pergunta
    resposta = handle_query(pergunta)

    return jsonify({"resposta": resposta}), 200


# executa o servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)