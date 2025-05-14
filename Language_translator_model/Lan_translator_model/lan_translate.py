import torch
from transformers import MarianMTModel, MarianTokenizer
import networkx as nx
import matplotlib.pyplot as plt

# Define translation paths (e.g., English -> French -> German -> Spanish)
translation_path = ['en', 'fr', 'de', 'es']

# Load models and tokenizers for each step in the path
def load_models(path):
    models = {}
    for i in range(len(path)-1):
        src, tgt = path[i], path[i+1]
        model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            models[(src, tgt)] = (tokenizer, model)
        except Exception as e:
            print(f"Model {model_name} not available: {e}")
    return models

# Translate text along the path
def translate_along_path(text, path, models):
    graph = nx.DiGraph()
    current_text = text
    graph.add_node(f"{path[0]}: {text}")

    for i in range(len(path) - 1):
        src, tgt = path[i], path[i+1]
        if (src, tgt) not in models:
            print(f"No model for {src} to {tgt}")
            break
        tokenizer, model = models[(src, tgt)]
        inputs = tokenizer(current_text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        current_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        graph.add_node(f"{tgt}: {current_text}")
        graph.add_edge(f"{path[i]}: {text if i == 0 else prev_text}", f"{tgt}: {current_text}")
        prev_text = current_text

    return graph

# Visualize the translation path
def draw_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=9, font_weight='bold')
    plt.title("Translation Path Learning")
    plt.show()

# Main function
if __name__ == "__main__":
    input_text = input("Enter text in English: ")
    models = load_models(translation_path)
    translation_graph = translate_along_path(input_text, translation_path, models)
    draw_graph(translation_graph)
