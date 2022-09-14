import torch
import numpy as np
from helper import *
from config.GlobalVariables import *
from SynthesisNetwork import SynthesisNetwork
from DataLoader import DataLoader
import convenience
import gradio as gr


def update_chosen_writers(writer1, writer2, weight, words, all_loaded_data):
    net.clamp_mdn = 0
    chosen_writers = [int(writer1.split(" ")[1]), int(writer2.split(" ")[1])]

    all_loaded_data = []
    for writer_id in chosen_writers:
        loaded_data = dl.next_batch(TYPE='TRAIN', uid=writer_id, tids=list(range(num_samples)))
        all_loaded_data.append(loaded_data)

    writer_mean_Ws = []
    for loaded_data in all_loaded_data:
        mean_global_W = convenience.get_mean_global_W(net, loaded_data, device)
        writer_mean_Ws.append(mean_global_W.detach())

    return gr.Slider.update(label=f"{writer1} vs. {writer2}"), chosen_writers, writer_mean_Ws, *update_writer_word(" ".join(words), writer_mean_Ws, all_loaded_data, weight)

def update_writer_word(target_word, writer_mean_Ws, all_loaded_data, writer_weight, device="cpu"):
    words = []
    for word in target_word.split(" "):
        if len(word) > 0:
            words.append(word)
        
    word_Ws = []
    word_Cs = []
    for word in words:
        writer_Ws, writer_Cs = convenience.get_DSD(net, word, writer_mean_Ws, all_loaded_data, device)
        word_Ws.append(writer_Ws)
        word_Cs.append(writer_Cs)

    if len(words) == 0:
        word_Ws.append(torch.tensor([]))
        word_Cs.append(torch.tensor([]))

    return words, word_Ws, word_Cs, *update_writer_slider(writer_weight, words, word_Ws, word_Cs)

def update_writer_slider(weight, words, all_word_Ws, all_word_Cs):
    weights = [1 - weight, weight]
    net.clamp_mdn = 0
    svg = convenience.draw_words_svg(words, all_word_Ws, all_word_Cs, weights, net)
    return gr.HTML.update(value=svg.tostring()), gr.File.update(visible=False), gr.Button.update(visible=True), weight, svg


def update_writer_download(writer_svg):
    writer_svg.saveas("./DSD_writer_interpolation.svg")
    return gr.File.update(value="./DSD_writer_interpolation.svg", visible=True), gr.Button.update(visible=False)

# for character blend
def update_blend_chars(c1, c2, weight, char_Ws):
    blend_chars = [c1, c2]
    char_Cs = torch.zeros(1, 2, convenience.L, convenience.L)
    for i in range(2):  # get corners of grid
        _, char_matrix = convenience.get_DSD(net, blend_chars[i], default_mean_global_W, [default_loaded_data], device)
        char_Cs[:, i, :, :] = char_matrix

    return gr.Slider.update(label=f"'{c1}' vs. '{c2}'"), char_Cs.detach(), blend_chars, *update_char_slider(weight, char_Ws, char_Cs, blend_chars)

def update_char_slider(weight, char_Ws, char_Cs, blend_chars):
    """Generates an image of handwritten text based on target_sentence"""
    net.clamp_mdn = 0
    character_weights = [1 - weight, weight]

    all_W_c = convenience.get_character_blend_W_c(character_weights, char_Ws, char_Cs)
    all_commands = convenience.get_commands(net, blend_chars[0], all_W_c)
    svg = convenience.commands_to_svg(all_commands, 750, 160, 375)
    return gr.HTML.update(value=svg.tostring()), gr.File.update(visible=False), gr.Button.update(visible=True), weight, svg

def update_char_download(char_svg):
    char_svg.saveas("./DSD_char_interpolation.svg")
    return gr.File.update(value="./DSD_char_interpolation.svg", visible=True), gr.Button.update(visible=False)

# for MDN
def update_mdn_word(target_word, scale_sd, clamp_mdn):
    mdn_words = []
    for word in target_word.split(" "):
        mdn_words.append(word)

    all_word_mdn_Ws = []
    all_word_mdn_Cs = []
    for word in mdn_words:
        all_writer_Ws, all_writer_Cs = convenience.get_DSD(net, word, default_mean_global_W, [default_loaded_data], device)
        all_word_mdn_Ws.append(all_writer_Ws)
        all_word_mdn_Cs.append(all_writer_Cs)

    return mdn_words, all_word_mdn_Ws, all_word_mdn_Cs, *sample_mdn(scale_sd, clamp_mdn, mdn_words, all_word_mdn_Ws, all_word_mdn_Cs)


def sample_mdn(maxs, maxr, mdn_words, all_word_mdn_Ws, all_word_mdn_Cs):
    net.clamp_mdn = maxr
    net.scale_sd = maxs
    svg = convenience.draw_words_svg(mdn_words, all_word_mdn_Ws, all_word_mdn_Cs, [1], net)
    return gr.HTML.update(value=svg.tostring()), gr.File.update(visible=False), gr.Button.update(visible=True), maxr, maxs, svg

def update_mdn_download(mdn_svg):
    mdn_svg.saveas("./DSD_add_randomness.svg")
    return gr.File.update(value="./DSD_add_randomness.svg", visible=True), gr.Button.update(visible=False)

device = 'cpu'
num_samples = 10

net = SynthesisNetwork(weight_dim=256, num_layers=3).to(device)

if not torch.cuda.is_available():
    net.load_state_dict(torch.load('./model/250000.pt', map_location=torch.device(device))["model_state_dict"])


dl = DataLoader(num_writer=1, num_samples=10, divider=5.0, datadir='./data/writers')

writer_options = [5, 14, 15, 16, 17, 22, 25, 80, 120, 137, 147, 151]
all_loaded_data_DEFAULT = []
chosen_writers_DEFAULT = [120, 80]
avail_char = "0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! ? \" ' * + - = : ; , . < > \ / [ ] ( ) # $ % &"
avail_char_list = avail_char.split(" ")
for writer_id in chosen_writers_DEFAULT:
    loaded_data = dl.next_batch(TYPE='TRAIN', uid=writer_id, tids=list(range(num_samples)))
    all_loaded_data_DEFAULT.append(loaded_data)

default_loaded_data = all_loaded_data_DEFAULT[-1]
default_mean_global_W = convenience.get_mean_global_W(net, default_loaded_data, device)

# data for writer interpolation
writer_words_DEFAULT = ["hello", "world"]
writer_mean_Ws_DEFAULT = []
writer_all_word_Ws_DEFAULT = []
writer_all_word_Cs_DEFAULT = []
writer_weight_DEFAULT = 0.7
writer_svg_DEFAULT = None

# data for char interpolation
char_chosen_DEFAULT = ["y", "s"]
char_mean_global_W_DEFAULT = None
char_weight_DEFAULT = 0.7
char_Ws_DEFAULT = default_mean_global_W.reshape(1, 1, convenience.L)
char_Cs_DEFAULT = None
char_svg_DEFAULT = None

# # data for MDN
mdn_words_DEFAULT = ["hello", "world"]
all_word_mdn_Ws_DEFAULT = None
all_word_mdn_Cs_DEFAULT = None
clamp_mdn_DEFAULT = 0.5
scale_sd_DEFAULT = 1
mdn_svg_DEFAULT = None

_wrds, writer_all_word_Ws_DEFAULT, writer_all_word_Cs_DEFAULT, _html, _file, _btn, _wt, _svg = update_writer_word(" ".join(writer_words_DEFAULT), writer_mean_Ws_DEFAULT, all_loaded_data_DEFAULT, writer_weight_DEFAULT)
_sldr, _wrtrs, writer_mean_Ws_DEFAULT, _wrds, _waww, _wawc, _html, _file, _btn, _wt, writer_svg_DEFAULT = update_chosen_writers(f"Writer {chosen_writers_DEFAULT[0]}", f"Writer {chosen_writers_DEFAULT[1]}", writer_weight_DEFAULT, writer_words_DEFAULT, all_loaded_data_DEFAULT)

_wrds, all_word_mdn_Ws_DEFAULT, all_word_mdn_Cs_DEFAULT, _html, _file, _btn, _maxr, _maxs, mdn_svg_DEFAULT = update_mdn_word(" ".join(mdn_words_DEFAULT), scale_sd_DEFAULT, clamp_mdn_DEFAULT)
_sldr, char_Cs_DEFAULT, _chrs, _html, _file, _btn, _wght, char_svg_DEFAULT = update_blend_chars(*char_chosen_DEFAULT, char_weight_DEFAULT, char_Ws_DEFAULT)

with gr.Blocks() as demo:
    all_loaded_data_var = gr.Variable(all_loaded_data_DEFAULT)
    chosen_writers_var = gr.Variable(chosen_writers_DEFAULT)
    # data for writer interpolation
    writer_words_var = gr.Variable(writer_words_DEFAULT)
    writer_mean_Ws_var = gr.Variable(writer_mean_Ws_DEFAULT)
    writer_all_word_Ws_var = gr.Variable([e.detach() for e in writer_all_word_Ws_DEFAULT])
    writer_all_word_Cs_var = gr.Variable([e.detach() for e in writer_all_word_Cs_DEFAULT])
    writer_weight_var = gr.Variable(writer_weight_DEFAULT)
    writer_svg_var = gr.Variable(writer_svg_DEFAULT)
    # data for char interpolation
    char_chosen_var = gr.Variable(char_chosen_DEFAULT)
    char_mean_global_W_var = gr.Variable(char_mean_global_W_DEFAULT)
    char_weight_var = gr.Variable(char_weight_DEFAULT)
    char_Ws_var = gr.Variable(char_Ws_DEFAULT.detach())
    char_Cs_var = gr.Variable(char_Cs_DEFAULT.detach())
    char_svg_var = gr.Variable(char_svg_DEFAULT)
    # # data for MDN
    mdn_words_var = gr.Variable(mdn_words_DEFAULT)
    all_word_mdn_Ws_var = gr.Variable([e.detach() for e in all_word_mdn_Ws_DEFAULT])
    all_word_mdn_Cs_var = gr.Variable([e.detach() for e in all_word_mdn_Cs_DEFAULT])
    clamp_mdn_var = gr.Variable(clamp_mdn_DEFAULT)
    scale_sd_var = gr.Variable(scale_sd_DEFAULT)
    mdn_svg_var = gr.Variable(mdn_svg_DEFAULT)

    with gr.Tabs():
        with gr.TabItem("Blend Writers"):
            target_word = gr.Textbox(label="Target Word", value=" ".join(writer_words_DEFAULT), max_lines=1)
            with gr.Row():
                left_ratio_options = ["Style " + str(id) for i, id in enumerate(writer_options) if i % 2 == 0]
                right_ratio_options = ["Style " + str(id) for i, id in enumerate(writer_options) if i % 2 == 1]
                with gr.Column():
                    writer1 = gr.Radio(left_ratio_options, value="Style 120", label="Style for first writer")
                with gr.Column():
                    writer2 = gr.Radio(right_ratio_options, value="Style 80", label="Style for second writer")
            with gr.Row():
                writer_slider = gr.Slider(0, 1, value=writer_weight_DEFAULT, label="Style 120 vs. Style 80")
            with gr.Row():
                writer_default_image = update_writer_slider(writer_weight_DEFAULT, writer_words_DEFAULT, writer_all_word_Ws_DEFAULT, writer_all_word_Cs_DEFAULT)
                writer_output = gr.HTML(writer_default_image[0]["value"])
            with gr.Row():
                writer_download_btn = gr.Button("Save to SVG file")
                writer_download_btn.style(full_width="true")
                writer_download = gr.File(interactive=False, show_label=False, visible=False)

            writer_slider.change(fn=update_writer_slider, 
                inputs=[writer_slider, writer_words_var, writer_all_word_Ws_var, writer_all_word_Cs_var], 
                outputs=[writer_output, writer_download, writer_download_btn, writer_weight_var, writer_svg_var], show_progress=False)
            target_word.submit(fn=update_writer_word, 
                inputs=[target_word, writer_mean_Ws_var, all_loaded_data_var, writer_weight_var], 
                outputs=[writer_words_var, writer_all_word_Ws_var, writer_all_word_Cs_var, writer_output, writer_download, writer_download_btn, writer_weight_var, writer_svg_var], show_progress=False)
            writer1.change(fn=update_chosen_writers, 
                inputs=[writer1, writer2, writer_weight_var, writer_words_var, all_loaded_data_var], 
                outputs=[writer_slider, chosen_writers_var, writer_mean_Ws_var, writer_words_var, writer_all_word_Ws_var, writer_all_word_Cs_var, writer_output, writer_download, writer_download_btn, writer_weight_var, writer_svg_var])
            writer2.change(fn=update_chosen_writers, 
                inputs=[writer1, writer2, writer_weight_var, writer_words_var, all_loaded_data_var], 
                outputs=[writer_slider, chosen_writers_var, writer_mean_Ws_var, writer_words_var, writer_all_word_Ws_var, writer_all_word_Cs_var, writer_output, writer_download, writer_download_btn, writer_weight_var, writer_svg_var])
            writer_download_btn.click(fn=update_writer_download, 
                inputs=[writer_svg_var], 
                outputs=[writer_download, writer_download_btn])

        with gr.TabItem("Blend Characters"):
            with gr.Row():
                with gr.Column():
                    char1 = gr.Dropdown(choices=avail_char_list, value=char_chosen_DEFAULT[0], label="Character 1")
                with gr.Column():
                    char2 = gr.Dropdown(choices=avail_char_list, value=char_chosen_DEFAULT[1], label="Character 2")
            with gr.Row():
                char_slider = gr.Slider(0, 1, value=char_weight_DEFAULT, label=f"'{char_chosen_DEFAULT[0]}' vs. '{char_chosen_DEFAULT[1]}'")
            with gr.Row():
                char_default_image = update_char_slider(char_weight_DEFAULT, char_Ws_DEFAULT, char_Cs_DEFAULT, char_chosen_DEFAULT)
                char_output = gr.HTML(char_default_image[0]["value"])
            with gr.Row():
                char_download_btn = gr.Button("Save to SVG file")
                char_download_btn.style(full_width="true")
                char_download = gr.File(interactive=False, show_label=False, visible=False)

            char_slider.change(fn=update_char_slider, 
                inputs=[char_slider, char_Ws_var, char_Cs_var, char_chosen_var], 
                outputs=[char_output, char_download, char_download_btn, char_weight_var, char_svg_var], show_progress=False)

            char1.change(fn=update_blend_chars, 
                inputs=[char1, char2, char_weight_var, char_Ws_var], 
                outputs=[char_slider, char_Cs_var, char_chosen_var, char_output, char_download, char_download_btn, char_weight_var, char_svg_var])
            char2.change(fn=update_blend_chars, 
                inputs=[char1, char2, char_weight_var, char_Ws_var], 
                outputs=[char_slider, char_Cs_var, char_chosen_var, char_output, char_download, char_download_btn, char_weight_var, char_svg_var])

            char_download_btn.click(fn=update_char_download, 
                inputs=[char_svg_var], 
                outputs=[char_download, char_download_btn], show_progress=True)

        with gr.TabItem("Add Randomness"):
            mdn_word = gr.Textbox(label="Target Word", value=" ".join(mdn_words_DEFAULT), max_lines=1)
            with gr.Row():
                with gr.Column():
                    max_rand = gr.Slider(0, 1, value=clamp_mdn_DEFAULT, label="Maximum Randomness")
                with gr.Column():
                    scale_rand = gr.Slider(0, 3, value=scale_sd_DEFAULT, label="Scale of Randomness")
            with gr.Row():
                mdn_sample_button = gr.Button(value="Resample")
            with gr.Row():
                default_im = sample_mdn(scale_sd_DEFAULT, clamp_mdn_DEFAULT, mdn_words_DEFAULT, all_word_mdn_Ws_DEFAULT, all_word_mdn_Cs_DEFAULT)
                mdn_output = gr.HTML(default_im[0]["value"])
            with gr.Row():
                randomness_download_btn = gr.Button("Save to SVG file")
                randomness_download = gr.File(interactive=False, show_label=False, visible=False)

            max_rand.change(fn=sample_mdn, 
                inputs=[scale_rand, max_rand, mdn_words_var, all_word_mdn_Ws_var, all_word_mdn_Cs_var], 
                outputs=[mdn_output, randomness_download, randomness_download_btn, clamp_mdn_var, scale_sd_var, mdn_svg_var], show_progress=False)
            scale_rand.change(fn=sample_mdn, 
                inputs=[scale_rand, max_rand, mdn_words_var, all_word_mdn_Ws_var, all_word_mdn_Cs_var], 
                outputs=[mdn_output, randomness_download, randomness_download_btn, clamp_mdn_var, scale_sd_var, mdn_svg_var], show_progress=False)
            mdn_sample_button.click(fn=sample_mdn, 
                inputs=[scale_rand, max_rand, mdn_words_var, all_word_mdn_Ws_var, all_word_mdn_Cs_var], 
                outputs=[mdn_output, randomness_download, randomness_download_btn, clamp_mdn_var, scale_sd_var, mdn_svg_var], show_progress=False)
                
            mdn_word.submit(fn=update_mdn_word, 
                inputs=[mdn_word, scale_sd_var, clamp_mdn_var], 
                outputs=[mdn_words_var, all_word_mdn_Ws_var, all_word_mdn_Cs_var, mdn_output, randomness_download, randomness_download_btn, clamp_mdn_var, scale_sd_var, mdn_svg_var], show_progress=False)

            randomness_download_btn.click(fn=update_mdn_download, 
                inputs=[mdn_svg_var],
                outputs=[randomness_download, randomness_download_btn])
            randomness_download_btn.style(full_width="true")
demo.launch()
