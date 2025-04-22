import gradio as gr # Library untuk membangun interface berbasis web
from rembg import remove # Digunakan untuk menghapus latar belakang dari gambar
from PIL import Image as PILImage, ImageFilter, Image # Untuk pemrosesan gambar
from io import BytesIO # Untuk menangani data byte
import requests # Untuk membuat permintaan HTTP
import cv2 # Untuk pemrosesan gambar
import numpy as np # Untuk operasi numerik
import webbrowser # Untuk membuka halaman web

def remove_and_replace_background(subject, background, blur_radius, replace_background, use_color_picker, color):
    with open(subject, 'rb') as subject_img_file:
        subject_img = subject_img_file.read() # Membuka subjek gambar file pada binary mode
    subject_no_bg = remove(subject_img, alpha_matting=True, alpha_matting_foreground_threshold=10) # Menghapus background dari gambar subjek menggunakan rembg library
    subject_img_no_bg = PILImage.open(BytesIO(subject_no_bg)).convert("RGBA") # Mengkonversi gambar dengan background yang sudah dihapus ke format RGBA
    
    if replace_background:
        if use_color_picker:
            background_img = PILImage.new("RGBA", subject_img_no_bg.size, color) # Jika color picker digunakan, maka akan membuat gambar baru dengan warna yang dipilih
        else:
            background_img = PILImage.open(background).convert("RGBA")
            background_img = background_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)) # Sebaliknya, akan membuka gambar background dan menerapkan efek gaussian blur
            background_img = background_img.resize(subject_img_no_bg.size) # Merubah ukuran background image sesuai dengan ukuran subjek gambar
        combined_img = PILImage.alpha_composite(background_img, subject_img_no_bg) # Menggabungkan gambar background dengan gambar subjek (tanpa background)
        combined_img.save("combined_image.png") # Menyimpan penggabungan gambar
        return "combined_image.png" # Kembali ke gambar gabungan yang disimpan
    else:
        subject_img_no_bg.save("subject_no_bg.png") # Jika tidak mengganti background, simpan gambar subjek dengan background yang sudah dihapus
        return "subject_no_bg.png" # Kembali ke gambar yang disimpan dengan latar belakang dihapus

def upscale_image(input_image_path, output_image_path, engine_id, api_key, api_host="https://api.stability.ai", width=None, height=None):
    with open(input_image_path, "rb") as file:
        image_data = file.read() # Buka file gambar input dalam mode biner

    headers = {
        "Accept": "image/png",
        "Authorization": f"Bearer {api_key}",
    } # Siapkan header untuk permintaan API

    files = {
        "image": image_data,
    } # Menyiapkan muatan file untuk permintaan API

    data = {} # Siapkan muatan data dengan lebar dan tinggi opsional
    if width:
        data["width"] = width
    if height:
        data["height"] = height

    response = requests.post( 
        f"{api_host}/v1/generation/{engine_id}/image-to-image/upscale",
        headers=headers,
        files=files,
        data=data
    ) # Membuat permintaan POST ke API untuk upscale gambar

    if response.status_code != 200:
        raise Exception(f"Non-200 response: {response.text}") # Periksa apakah kode status respons bukan 200 (OK)

    try:
        nparr = np.frombuffer(response.content, np.uint8) # Mengonversi konten respons menjadi NumPy array
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Mengencode numpy array menjadi gambar
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Mengonversi gambar dari BGR (format OpenCV) ke RGB
    except Exception as e:
        raise Exception(f"Invalid image data: {e}") # Memunculkan pengecualian jika data gambar tidak valid

    cv2.imwrite(output_image_path, img_np) # Menyimpan gambar yang ditingkatkan ke jalur output yang ditentukan
    return output_image_path # Kembali ke jalur gambar keluaran

def upscale_gradio(input_image):
    output_image_path = "upscaled_image.png" # Menetapkan path gambar output
    input_image_path = "input_image.png" # Menetapkan path gambar input

    if np.max(input_image) > 1: # Periksa apakah nilai piksel gambar input lebih besar dari 1 (dengan asumsi gambar mungkin berada dalam kisaran 0-1)
        cv2.imwrite(input_image_path, np.array(input_image)) # Menyimpan gambar input ke file jika nilai piksel berada dalam kisaran 0-255
    else:
        cv2.imwrite(input_image_path, np.array(input_image) * 255) # Jika nilai piksel berada dalam kisaran 0-1, skala ke 0-255 dan simpan ke file

    upscale_image(input_image_path, output_image_path, "esrgan-v1-x2plus", "sk-snxMfG2LVsLyezE46G9GSxgEBMy9a2rBVsIBQWCrd3n6L5pP", width=1024) # Panggil fungsi upscale_image untuk meningkatkan kualitas gambar
    return output_image_path # Kembali pada path ke gambar yang ditingkatkan

def gray(input_img):
    image_path = 'image_gray.png' # Menentukan jalur tempat gambar grayscale akan disimpan
    image = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY) # Mengonversi gambar input ke grayscale menggunakan OpenCV
    cv2.imwrite(image_path, image) # Menyimpan gambar skala abu-abu ke jalur yang ditentukan
    return image_path # Kembali pada path ke penyimpanan gambar grayscale

def adjust_brightness_and_darkness(input_img, brightness_enabled, brightness_value, darkness_enabled, darkness_value):
    image = input_img.copy() # Membuat salinan gambar masukan untuk menghindari modifikasi gambar aslinya

    if brightness_enabled: # Sesuaikan kecerahan jika fitur kecerahan diaktifkan
        mat = np.ones(image.shape, dtype='uint8') * brightness_value # Buat matriks dengan bentuk yang sama seperti gambar, diisi dengan nilai kecerahan
        image = cv2.add(image, mat) # Add the brightness matrix to the image

    if darkness_enabled: # Adjust darkness if the darkness feature is enabled
        mat = np.ones(image.shape, dtype='uint8') * darkness_value # Create a matrix of the same shape as the image, filled with the darkness value
        image = cv2.subtract(image, mat) # Subtract the darkness matrix from the image

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Mengonversi gambar dari ruang warna BGR ke ruang warna RGB
    image_path = 'adjusted_image.png' # Menetapkan jalur tempat gambar yang sudah disesuaikan akan disimpan
    cv2.imwrite(image_path, image_rgb) # Menyimpan gambar yang disesuaikan ke jalur yang ditentukan
    return image_path # Kembali pada path ke gambar yang disesuaikan yang disimpan

def rotate_image(img_input, degrees):
    image_path = 'rotated.png' # Menentukan nama file yang akan disimpan
    height, width = img_input.shape[:2] # Dapatkan tinggi dan lebar gambar masukan
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1) # Hitung matriks rotasi untuk sudut yang diberikan di sekeliling bagian tengah gambar
    rotated_image = cv2.warpAffine(img_input, rotation_matrix, (width, height)) # Menerapkan rotasi ke gambar dengan menggunakan fungsi warpAffine
    rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB) # Convert the rotated image from BGR color space to RGB color space
    cv2.imwrite(image_path, rotated_image_rgb) # Menyimpan gambar yang diputar ke path yang ditentukan
    return image_path # Kembali pada path ke gambar yang diputar yang disimpan

def generate_iopaint_link():
    return "https://huggingface.co/spaces/Pontarids/IOPaint_Runner" # Berisi link deploy link untuk tool IOPaint

def skew_image(image, horizontal_skew, vertical_skew):
    image_np = np.array(image) # Mengonversi gambar input ke NumPy array
    rows, cols, ch = image_np.shape # Dapatkan dimensi gambar

    horizontal_factor = horizontal_skew / 100.0 # Hitung faktor skew horizontal 
    vertical_factor = vertical_skew / 100.0 # Hitung faktor skew vertikal

    M_horizontal = np.float32([[1, horizontal_factor, 0],
                               [0, 1, 0],
                               [0, 0, 1]]) # Tentukan matriks transformasi untuk skew horizontal

    M_vertical = np.float32([[1, 0, 0],
                             [vertical_factor, 1, 0],
                             [0, 0, 1]]) # Tentukan matriks transformasi untuk skew horizontal

    skewed_image_horizontal = cv2.warpPerspective(image_np, M_horizontal, (cols, rows)) # Menerapkan transformasi skew horizontal
    skewed_image_vertical = cv2.warpPerspective(skewed_image_horizontal, M_vertical, (cols, rows)) # Menerapkan transformasi kemiringan vertikal pada gambar skew secara horizontal

    skewed_image = Image.fromarray(skewed_image_vertical) # Ubah NumPy array yang dihasilkan kembali menjadi Gambar PIL
    return skewed_image # Kembali ke gambar skew

with gr.Blocks() as demo: # Untuk dapat menggunakan gr.Tab diperlukan gr.Blocks

    with gr.Tab("Remove and Replace Background"): # Membuat tab bertuliskan "Remove and Replace Background"
        subject_img_input = gr.Image(type="filepath") # Input subjek image
        background_img_input = gr.Image(type="filepath") # Input custom background image
        blur_radius_slider = gr.Slider(0, 100, label="Blur Radius") # Slider yang digunakan untuk mengontrol blur
        replace_bg_checkbox = gr.Checkbox(label="Replace Background") # Checkbox untuk mengaktifkan fitur custom background
        use_color_picker_checkbox = gr.Checkbox(label="Use Color Picker") # Checkbox untuk mengaktifkan fitur custom warna solid background
        color_picker = gr.ColorPicker(label="Background Color") # Digunakan untuk memilih warna
        processed_img_output = gr.Image() # Output gambar setelah diedit
        submit_button = gr.Button("Submit") # Button submit untuk memulai proses
        submit_button.click(remove_and_replace_background, inputs=[subject_img_input, background_img_input, blur_radius_slider, replace_bg_checkbox, use_color_picker_checkbox, color_picker], outputs=processed_img_output) # Memasukan parameter input dan output untuk sebagai proses pada submit button
    
    with gr.Tab("Upscale Image"): # Membuat tab bertuliskan "Upscale Image"
        img_input_upscale = gr.Image() # Input dari gambar yang ingin diupscale
        img_output_upscale = gr.Image() # Output dari gambar yang sudah diupscale
        img_button_upscale = gr.Button("Submit") # Button submit untuk memulai proses
        img_button_upscale.click(upscale_gradio, inputs=img_input_upscale, outputs=img_output_upscale) # Memasukan parameter input dan output untuk sebagai proses pada submit button
    
    with gr.Tab("Gray"): # Membuat tab bertuliskan "Gray"
        img_input_gray = gr.Image() # Input dari gambar yang ingin di ubah ke grayscale
        img_output_gray = gr.Image() # Output dari gambar yang sudah di ubah ke grayscale
        img_button_gray = gr.Button("Submit") # Button submit untuk memulai proses
        img_button_gray.click(gray, inputs=img_input_gray, outputs=img_output_gray) # Memasukan parameter input dan output untuk sebagai proses pada submit button
    
    with gr.Tab("Brightness and Darkness"): # Membuat tab bertuliskan "Brightness and Darkness"
        img_input_contrast = gr.Image() # Input dari gambar yang ingin di ubah brightness dan darknessnya
        brightness_checkbox = gr.Checkbox(label="Enable Brightness Adjustment") # Checkbox untuk mengaktifkan fitur brightness adjustment
        brightness_slider = gr.Slider(0, 255, label="Brightness Value") # Slider untuk mengatur seberapa banyak brightness yang diinginkan
        darkness_checkbox = gr.Checkbox(label="Enable Darkness Adjustment") # Checkbox untuk mengaktifkan fitur darkness adjustment
        darkness_slider = gr.Slider(0, 255, label="Darkness Value") # Slider untuk mengatur seberapa banyak darkness yang diinginkan
        img_output_contrast = gr.Image() # Output dari gambar yang telah di ubah brightness dan darknessnya
        img_button_contrast = gr.Button("Submit") # Button submit untuk memulai proses
        img_button_contrast.click(adjust_brightness_and_darkness, inputs=[img_input_contrast, brightness_checkbox, brightness_slider, darkness_checkbox, darkness_slider], outputs=img_output_contrast) # Memasukan parameter input dan output untuk sebagai proses pada submit button
    
    with gr.Tab("Rotate Image"): # Membuat tab bertuliskan "Rotate Image"
        temp_slider = gr.Slider(minimum=0, maximum=360, interactive=True, label="Slide me") # Slider untuk mengatur berapa derajat gambar ingin diputar
        img_input_rotate = gr.Image() # Input dari gambar yang ingin di ubah posisi rotasinya
        img_output_rotate = gr.Image() # Output dari gambar yang telah di ubah posisi rotasinya
        img_button_rotate = gr.Button("Submit") # Button submit untuk memulai proses
        img_button_rotate.click(rotate_image, inputs=[img_input_rotate, temp_slider], outputs=img_output_rotate) # Memasukan parameter input dan output untuk sebagai proses pada submit button

    with gr.Tab("Skew Tool"): # Membuat tab bertuliskan "Skew Tool"
        image_input = gr.Image(type="pil", label="Upload an image") # Input dari gambar yang ingin diberi efek skew
        horizontal_slider = gr.Slider(minimum=-100, maximum=100, value=0, label="Horizontal Skew") # Slider untuk mengatur horizontal skew
        vertical_slider = gr.Slider(minimum=-100, maximum=100, value=0, label="Vertical Skew") # Slider untuk mengatur vertikal skew
        output_image = gr.Image(type="pil", label="Skewed Image") # Output dari gambar yang ingin diberi efek skew
        submit_button_skew = gr.Button("Submit") # Button submit untuk memulai proses
        submit_button_skew.click(skew_image, inputs=[image_input, horizontal_slider, vertical_slider], outputs=output_image) # Memasukan parameter input dan output untuk sebagai proses pada submit button

    with gr.Tab("Object Remover 👑"):
        link_output = gr.Markdown() # Output akan berupa link menuju pada IOPaint tool
        link_button = gr.Button("Generate IOPaint Link") # Button untuk membuat link ke IOPaint tool
        link_button.click(generate_iopaint_link, outputs=link_output) # Memasukan parameter input dan output untuk sebagai proses pada Generate IOPaint Link button
    
demo.launch(share=True) # share=True ini akan membuat link yang dapat diakses oleh publik
