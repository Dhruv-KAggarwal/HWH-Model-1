from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import random
import csv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load BERT model and tokenizer
model_path = 'e:/Hackathon/HWH/bert/model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def read_fasta(file_path):
    sequences = []
    try:
        with open(file_path, 'r') as file:
            sequence = ""
            for line in file:
                if line.startswith(">"):
                    if sequence:
                        sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                sequences.append(sequence)
    except (OSError, IOError) as e:
        print(f"Error reading FASTA file: {e}")
    return sequences

def read_csv(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except (OSError, IOError) as e:
        print(f"Error reading CSV file: {e}")
    return data

def validate_sequence(sequence):
    return all(char in 'ATGC' for char in sequence)

def get_random_sequence(sequences):
    return random.choice(sequences)

def get_random_entry(data):
    return random.choice(data)

def generate_pdf(data, file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 100, "Predicted Data:")
    y = height - 150
    for key, value in data.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 20
    c.save()

def predict(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    global random_entry
    random_sequence = None
    random_entry = None
    if request.method == 'POST':
        input_sequence = request.form.get('sequence', '').upper()
        file = request.files.get('file')

        if input_sequence and validate_sequence(input_sequence):
            sequences = [input_sequence]
        elif file and file.filename.endswith(('.fasta', '.fa')):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            sequences = read_fasta(file_path)
        else:
            sequences = []

        if sequences:
            random_sequence = get_random_sequence(sequences)
            prediction = predict(random_sequence)
            file_path = 'e:/Hackathon/HWH/synthetic_dna_beauty_dataset.csv'
            data = read_csv(file_path)
            if data:
                random_entry = data[prediction]
            else:
                return "No data found in the file."
        else:
            return "No sequences found."

    return render_template('index.html', random_sequence=random_sequence, random_entry=random_entry)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    global random_entry
    if random_entry:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_data.pdf')
        generate_pdf(random_entry, pdf_path)
        return send_file(pdf_path, as_attachment=True)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
