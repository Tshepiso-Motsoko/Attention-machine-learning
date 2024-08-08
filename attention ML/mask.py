import sys
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200

def main():
    text = input("Text: ")

    # Tokenize input and look for the [MASK] token
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use the BERT model to predict the masked word and extract attention scores
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Process predictions and print them
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions for each attention head
    visualize_attentions(inputs.tokens(), result.attentions)

def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    # Convert the inputs to a list of IDs and search for the mask token ID
    input_ids = inputs["input_ids"].numpy().flatten()
    return int(np.where(input_ids == mask_token_id)[0][0]) if mask_token_id in input_ids else None

def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    # Convert the attention score to a value between 0 and 255 (for RGB)
    gray_value = int(attention_score * 255)
    return (gray_value, gray_value, gray_value)

def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.
    """
    # Loop through each layer and head to generate attention diagrams
    for i, layer in enumerate(attentions):
        for j, head in enumerate(layer[0]):
            generate_diagram(i + 1, j + 1, tokens, head)

def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head.
    """
    # Diagram setup and token placement
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw tokens and attention weights
    for i, token in enumerate(tokens):
        # Token column drawing
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text((image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
                        token, fill="white", font=FONT)
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Token row drawing
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text((PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
                  token, fill="white", font=FONT)

        # Draw the attention grid
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            y = PIXELS_PER_WORD + i * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j].numpy())
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save the generated diagram
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")

if __name__ == "__main__":
    main()
