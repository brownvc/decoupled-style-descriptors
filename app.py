import torch
import numpy as np
from helper import *
from config.GlobalVariables import *
from SynthesisNetwork import SynthesisNetwork
from DataLoader import DataLoader
import convenience
import gradio as gr

device = 'cpu'
num_samples = 10

net = SynthesisNetwork(weight_dim=256, num_layers=3).to(device)

if not torch.cuda.is_available():
    net.load_state_dict(torch.load('./model/250000.pt', map_location=torch.device(device))["model_state_dict"])


dl = DataLoader(num_writer=1, num_samples=10, divider=5.0, datadir='./data/writers')


writer_options = [5, 14, 15, 16, 17, 22, 25, 80, 120, 137, 147, 151]
all_loaded_data = []
chosen_writers = [120, 80]
avail_char = "0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! ? \" ' * + - = : ; , . < > \ / [ ] ( ) # $ % &"
avail_char_list = avail_char.split(" ")
for writer_id in chosen_writers:
    loaded_data = dl.next_batch(TYPE='TRAIN', uid=writer_id, tids=list(range(num_samples)))
    all_loaded_data.append(loaded_data)

default_loaded_data = all_loaded_data[-1]

# data for writer interpolation
writer_words = ["hello", "world"]
writer_mean_Ws = []
all_word_writer_Ws = []
all_word_writer_Cs = []
writer_weight = 0.7
writer_svg = None

# data for char interpolation
blend_chars = ["y", "s"]
char_mean_global_W = None
char_weight = 0.7
default_mean_global_W = convenience.get_mean_global_W(net, default_loaded_data, device)
char_Ws = default_mean_global_W.reshape(1, 1, convenience.L)
char_Cs = all_Cs = torch.zeros(1, 2, convenience.L, convenience.L)
char_svg = None

# data for MDN
mdn_words = ["hello", "world"]
mdn_mean_global_W = None
all_word_mdn_Ws = []
all_word_mdn_Cs = []
mdn_svg = None

def update_writer_word(target_word):
    writer_words.clear()
    for word in target_word.split(" "):
        writer_words.append(word)

    all_word_writer_Ws.clear()
    all_word_writer_Cs.clear()
    for word in writer_words:
        all_writer_Ws, all_writer_Cs = convenience.get_DSD(net, word, writer_mean_Ws, all_loaded_data, device)
        all_word_writer_Ws.append(all_writer_Ws)
        all_word_writer_Cs.append(all_writer_Cs)

    return update_writer_slider(writer_weight)


# for writer interpolation
def update_writer_slider(val):
    global writer_weight
    global writer_svg
    writer_weight = val
    weights = [1 - writer_weight, writer_weight]

    net.clamp_mdn = 0
    writer_svg = convenience.draw_words_svg(writer_words, all_word_writer_Ws, all_word_writer_Cs, weights, net)
    return gr.HTML.update(value=writer_svg.tostring()), gr.Slider.update(visible=False), gr.Button.update(visible=True)


def update_chosen_writers(writer1, writer2):
    net.clamp_mdn = 0
    chosen_writers[0], chosen_writers[1] = int(writer1.split(" ")[1]), int(writer2.split(" ")[1])

    all_loaded_data.clear()
    for writer_id in chosen_writers:
        loaded_data = dl.next_batch(TYPE='TRAIN', uid=writer_id, tids=list(range(num_samples)))
        all_loaded_data.append(loaded_data)

    writer_mean_Ws.clear()
    for loaded_data in all_loaded_data:
        mean_global_W = convenience.get_mean_global_W(net, loaded_data, device)
        writer_mean_Ws.append(mean_global_W)

    return gr.Slider.update(label=f"{writer1} vs. {writer2}"), *update_writer_slider(writer_weight)

def update_writer_download():
    writer_svg.saveas("./DSD_writer_interpolation.svg")
    return gr.File.update(value="./DSD_writer_interpolation.svg", visible=True), gr.Button.update(visible=False)

# for character blend

def update_char_slider(weight):
    """Generates an image of handwritten text based on target_sentence"""
    global char_weight
    global char_svg

    net.clamp_mdn = 0

    char_weight = weight
    character_weights = [1 - weight, weight]

    all_W_c = convenience.get_character_blend_W_c(character_weights, char_Ws, char_Cs)
    all_commands = convenience.get_commands(net, blend_chars[0], all_W_c)
    char_svg = convenience.commands_to_svg(all_commands, 750, 160, 375)
    return gr.HTML.update(value=char_svg.tostring()), gr.Slider.update(visible=False), gr.Button.update(visible=True)


def update_blend_chars(c1, c2):
    global blend_chars
    blend_chars[0], blend_chars[1] = c1, c2

    for i in range(2):  # get corners of grid
        _, char_matrix = convenience.get_DSD(net, blend_chars[i], default_mean_global_W, [default_loaded_data], device)
        char_Cs[:, i, :, :] = char_matrix

    return gr.Slider.update(label=f"'{c1}' vs. '{c2}'")

def update_char_download():
    char_svg.saveas("./DSD_char_interpolation.svg")
    return gr.File.update(value="./DSD_char_interpolation.svg", visible=True), gr.Button.update(visible=False)

# for MDN


def update_mdn_word(target_word):
    mdn_words.clear()
    for word in target_word.split(" "):
        mdn_words.append(word)

    all_word_mdn_Ws.clear()
    all_word_mdn_Cs.clear()
    for word in mdn_words:
        all_writer_Ws, all_writer_Cs = convenience.get_DSD(net, word, default_mean_global_W, [default_loaded_data], device)
        all_word_mdn_Ws.append(all_writer_Ws)
        all_word_mdn_Cs.append(all_writer_Cs)

    return sample_mdn(net.scale_sd, net.clamp_mdn)


def sample_mdn(maxs, maxr):
    global mdn_svg
    net.clamp_mdn = maxr
    net.scale_sd = maxs
    mdn_svg = convenience.draw_words_svg(mdn_words, all_word_mdn_Ws, all_word_mdn_Cs, [1], net)
    return gr.HTML.update(value=mdn_svg.tostring()), gr.Slider.update(visible=False), gr.Button.update(visible=True)

def update_mdn_download():
    mdn_svg.saveas("./DSD_add_randomness.svg")
    return gr.File.update(value="./DSD_add_randomness.svg", visible=True), gr.Button.update(visible=False)

update_writer_word(" ".join(writer_words))
update_chosen_writers(f"Writer {chosen_writers[0]}", f"Writer {chosen_writers[1]}")

update_mdn_word(" ".join(writer_words))
update_blend_chars(*blend_chars)

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Blend Writers"):
            target_word = gr.Textbox(label="Target Word", value=" ".join(writer_words), max_lines=1)
            with gr.Row():
                left_ratio_options = ["Style " + str(id) for i, id in enumerate(writer_options) if i % 2 == 0]
                right_ratio_options = ["Style " + str(id) for i, id in enumerate(writer_options) if i % 2 == 1]
                with gr.Column():
                    writer1 = gr.Radio(left_ratio_options, value="Style 120", label="Style for first writer")
                with gr.Column():
                    writer2 = gr.Radio(right_ratio_options, value="Style 80", label="Style for second writer")
            with gr.Row():
                writer_submit = gr.Button("Submit")
            with gr.Row():
                writer_slider = gr.Slider(0, 1, value=writer_weight, label="Style 120 vs. Style 80")
            with gr.Row():
                writer_default_image = update_writer_slider(writer_weight)
                writer_output = gr.HTML(writer_default_image[0]["value"])
            with gr.Row():
                writer_download_btn = gr.Button("Save to SVG file")
                writer_download = gr.File(interactive=False, show_label=False, visible=False)
            writer_submit.click(fn=update_writer_slider, inputs=[writer_slider], outputs=[writer_output, writer_download, writer_download_btn], show_progress=False)
            writer_slider.change(fn=update_writer_slider, inputs=[writer_slider], outputs=[writer_output, writer_download, writer_download_btn], show_progress=False)
            target_word.submit(fn=update_writer_word, inputs=[target_word], outputs=[writer_output, writer_download, writer_download_btn], show_progress=False)

            writer1.change(fn=update_chosen_writers, inputs=[writer1, writer2], outputs=[writer_slider, writer_output, writer_download, writer_download_btn])
            writer2.change(fn=update_chosen_writers, inputs=[writer1, writer2], outputs=[writer_slider, writer_output, writer_download, writer_download_btn])
            writer_download_btn.click(fn=update_writer_download, inputs=[], outputs=[writer_download, writer_download_btn])
            writer_download_btn.style(full_width="true")
        with gr.TabItem("Blend Characters"):
            with gr.Row():
                with gr.Column():
                    char1 = gr.Dropdown(choices=avail_char_list, value=blend_chars[0], label="Character 1")
                with gr.Column():
                    char2 = gr.Dropdown(choices=avail_char_list, value=blend_chars[1], label="Character 2")
            with gr.Row():
                char_submit_button = gr.Button(value="Submit")
            with gr.Row():
                char_slider = gr.Slider(0, 1, value=char_weight, label=f"'{blend_chars[0]}' vs. '{blend_chars[1]}'")
            with gr.Row():
                char_default_image = update_char_slider(char_weight)
                char_output = gr.HTML(char_default_image[0]["value"])
            with gr.Row():
                char_download_btn = gr.Button("Save to SVG file")
                char_download = gr.File(interactive=False, show_label=False, visible=False)

            char_slider.change(fn=update_char_slider, inputs=[char_slider], outputs=[char_output, char_download, char_download_btn], show_progress=False)

            char1.change(fn=update_blend_chars, inputs=[char1, char2], outputs=[char_slider])
            char2.change(fn=update_blend_chars, inputs=[char1, char2], outputs=[char_slider])

            char_submit_button.click(fn=update_char_slider, inputs=[char_slider], outputs=[char_output, char_download, char_download_btn], show_progress=False)

            char_download_btn.click(fn=update_char_download, inputs=[], outputs=[char_download, char_download_btn], show_progress=True)
            char_download_btn.style(full_width="true")
        with gr.TabItem("Add Randomness"):
            mdn_word = gr.Textbox(label="Target Word", value=" ".join(mdn_words), max_lines=1)
            with gr.Row():
                with gr.Column():
                    max_rand = gr.Slider(0, 1, value=net.clamp_mdn, label="Maximum Randomness")
                with gr.Column():
                    scale_rand = gr.Slider(0, 3, value=net.scale_sd, label="Scale of Randomness")
            with gr.Row():
                mdn_sample_button = gr.Button(value="Resample")
            with gr.Row():
                default_im = sample_mdn(net.scale_sd, net.clamp_mdn)
                mdn_output = gr.HTML(default_im[0]["value"])
            with gr.Row():
                randomness_download_btn = gr.Button("Save to SVG file")
                randomness_download = gr.File(interactive=False, show_label=False, visible=False)

            max_rand.change(fn=sample_mdn, inputs=[scale_rand, max_rand], outputs=[mdn_output, randomness_download, randomness_download_btn], show_progress=False)
            scale_rand.change(fn=sample_mdn, inputs=[scale_rand, max_rand], outputs=[mdn_output, randomness_download, randomness_download_btn], show_progress=False)
            mdn_sample_button.click(fn=sample_mdn, inputs=[scale_rand, max_rand], outputs=[mdn_output, randomness_download, randomness_download_btn], show_progress=False)
            mdn_word.submit(fn=update_mdn_word, inputs=[mdn_word], outputs=[mdn_output, randomness_download, randomness_download_btn], show_progress=False)

            randomness_download_btn.click(fn=update_mdn_download, inputs=[], outputs=[randomness_download, randomness_download_btn])
            randomness_download_btn.style(full_width="true")
demo.launch()
