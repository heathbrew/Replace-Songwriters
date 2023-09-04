from flask import Flask, render_template, request, redirect, send_file, url_for
import csv
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import random
import nltk
import openai
from nltk.corpus import cmudict
from tensorflow.keras.utils import pad_sequences
from sklearn.preprocessing import StandardScaler
import textblob
import enchant
from autocorrect import Speller

app = Flask(__name__)

# Load the model and tokenizer
model = tf.keras.models.load_model('model_SongAI_save.tf')
with open('tokenizerSongAI.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Load the Standard Scaler
with open('scalerSongAI.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the maximum sequence length
max_sequence_length = 100

# Function to generate merged text
def generate_merged_text(seed_text, num_of_words, artist, genre, year):
    seed_tokens = tokenizer.texts_to_sequences([seed_text + artist + genre + year])
    seed_tokens = pad_sequences(seed_tokens, maxlen=max_sequence_length)

    numerical_inputs = scaler.transform([[year, num_of_words]])

    generated_text = seed_text

    for _ in range(num_of_words):
        prediction = model.predict([seed_tokens, numerical_inputs])
        prediction = prediction[0][-1]

        prediction /= np.sum(prediction)  # Normalize probabilities

        valid_indices = [idx for word, idx in tokenizer.word_index.items() if idx < len(prediction)]
        output_word_index = np.random.choice(valid_indices, p=prediction[valid_indices] / np.sum(prediction[valid_indices]))
        output_word = tokenizer.index_word.get(output_word_index, '')

        generated_text += ' ' + output_word

        seed_tokens = np.append(seed_tokens[:, 1:], [[output_word_index]], axis=1)

    return generated_text

# Function to correct spelling
def correct_spelling(paragraph):
    # Initialize the spell checker
    spell = Speller(lang='en')

    # Use textblob for initial spell checking
    blob = textblob.TextBlob(paragraph)
    corrected_paragraph = blob.correct()

    checked_paragraph = []
    for word in corrected_paragraph.words:
        # Check if the word is spelled correctly
        corrected_word = spell(word)
        checked_paragraph.append(corrected_word)

    return " ".join(checked_paragraph)

# Function to generate lyrics using Markov chain
def generate_lyrics(input_lyrics):
    # Create a list of words from the input lyrics
    words = input_lyrics.split(' ')

    # Create a Markov chain model
    markov_model = {}
    for i in range(1, len(words)):
        if words[i-1] not in markov_model:
            # If the word is not already in the model, add it
            markov_model[words[i-1]] = [words[i]]
        else:
            # If the word is already in the model, append the following word to the list
            markov_model[words[i-1]].append(words[i])

    # Choose a random word from the input lyrics to start the new lyrics
    current_word = random.choice(list(markov_model.keys()))
    new_lyrics = current_word.capitalize()

    # Generate the lyrics
    for i in range(len(words)-1):  # Subtract 1 because we already added the first word
        if current_word not in markov_model:
            break
        next_word = random.choice(markov_model[current_word])
        new_lyrics += ' ' + next_word
        current_word = next_word

    return new_lyrics

# Function to capitalize lines in the lyrics
def capitalize_lines(text):
    lines = text.split("\n")
    capitalized_lines = [line.capitalize() for line in lines]
    return "\n".join(capitalized_lines)

# Function to generate a verse
def generate_verse(lyrics):
    verse = []
    verse.append(lyrics)
    verse.append("")
    return "\n".join(verse)

# Function to generate a chorus
def generate_chorus(lyrics):
    chorus = []
    chorus.append(lyrics)
    chorus.append("")
    return "\n".join(chorus)

# Function to generate a bridge
def generate_bridge(lyrics):
    bridge = []
    bridge.append(lyrics)
    bridge.append("")
    return "\n".join(bridge)

# Function to generate a song based on user inputs
def generate_song(words, rhyme_scheme, verse_length, chorus_length, bridge_length):
    song_parts = []
    lyrics_idx = 0
    rhyme_idx = 0

    while lyrics_idx < len(words):
        if lyrics_idx + verse_length <= len(words) and rhyme_scheme[rhyme_idx] == 'A':
            verse_lyrics = " ".join(words[lyrics_idx:lyrics_idx + verse_length])
            song_parts.append(generate_verse(verse_lyrics))
            lyrics_idx += verse_length
        elif lyrics_idx + chorus_length <= len(words) and rhyme_scheme[rhyme_idx] == 'B':
            chorus_lyrics = " ".join(words[lyrics_idx:lyrics_idx + chorus_length])
            song_parts.append(generate_chorus(chorus_lyrics))
            lyrics_idx += chorus_length
        elif lyrics_idx + bridge_length <= len(words) and rhyme_scheme[rhyme_idx] == 'C' and bridge_length > 0:
            bridge_lyrics = " ".join(words[lyrics_idx:lyrics_idx + bridge_length])
            song_parts.append(generate_bridge(bridge_lyrics))
            lyrics_idx += bridge_length
        else:
            break  # add a break condition in case we can't increment lyrics_idx

        rhyme_idx = (rhyme_idx + 1) % len(rhyme_scheme)

        song = "\n".join(song_parts)
    return song


def check_credentials(username, password):
    with open('credentials.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row[0] == username and row[1] == password:
                return True
    return False


from apikeys import chatgpt
openai.api_key = chatgpt()

def generate_song_lyrics(user_input):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "System prompt..."},
            {"role": "user", "content": user_input}
        ]
    )
    response = completion.choices[0].message['content']
    return response

@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/dashboard')
def content():
    return render_template('dashboard.html')

@app.route('/Song')
def home():
    return render_template('Song.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if check_credentials(username, password):
            return redirect(url_for('content'))
        else:
            error = 'Invalid username or password.'
    return render_template('signin.html', error=error)

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Retrieve the form data and generate the song
        seed_text = request.form['seed_text']
        num_of_words = int(request.form['num_of_words'])
        artist = request.form['artist']
        genre = request.form['genre']
        year = request.form['year']
        # verse_length = int(request.form['verse_length'])
        # chorus_length = int(request.form['chorus_length'])
        # bridge_length = int(request.form['bridge_length'])
        # rhyme_scheme = request.form['rhyme_scheme']
        generated_music = generate_merged_text(seed_text, num_of_words, artist, genre, year)
        corrected_paragraph = correct_spelling(generated_music)
        # markov_song = capitalize_lines(generate_lyrics(corrected_paragraph))
        # generated_song = generate_song(markov_song.split(), rhyme_scheme, verse_length, chorus_length, bridge_length)


        return render_template('result_actual.html', generated_text=corrected_paragraph)
        # return render_template('result_actual.html', generated_song=generated_song)
        # return render_template('result_actual.html', generated_song=generated_song, generated_text=generated_music, new_lyrics=markov_song,
        #                        verse_length=verse_length, chorus_length=chorus_length, bridge_length=bridge_length, rhyme_scheme=rhyme_scheme)
    else:
        # Render the initial form for generating song
        return render_template('Song.html', generated_text=None, generated_song=None, new_lyrics=None)

# Existing routes and functions...

# Add a new route to generate Markov music separately
@app.route('/generate-markov', methods=['GET', 'POST'])
def generate_markov():
    generated_text = request.args.get('generated_text')  # Retrieve the generated text from the query parameters
    # verse_length = int(request.args.get('verse_length'))
    # chorus_length = int(request.args.get('chorus_length'))
    # bridge_length = int(request.args.get('bridge_length'))
    # rhyme_scheme = request.args.get('rhyme_scheme')

    corrected_paragraph = correct_spelling(generated_text)
    markov_song = capitalize_lines(generate_lyrics(corrected_paragraph))

    # return render_template('result_actual.html', generated_text=generated_text, new_lyrics=markov_song,
    #                        verse_length=verse_length, chorus_length=chorus_length, bridge_length=bridge_length, rhyme_scheme=rhyme_scheme)
    return render_template('result_actual.html', generated_text=generated_text, new_lyrics=markov_song)

# Add a new route to regenerate the final song separately
@app.route('/regenerate_song', methods=['GET', 'POST'])
def regenerate_song():
    generated_text = request.args.get('generated_text')
    verse_length = int(request.args.get('verse_length'))
    chorus_length = int(request.args.get('chorus_length'))
    bridge_length = int(request.args.get('bridge_length'))
    rhyme_scheme = request.args.get('rhyme_scheme')
    markov_song = request.args.get('new_lyrics')
    generated_song = generate_song(markov_song.split(), rhyme_scheme, verse_length, chorus_length, bridge_length)

    return render_template('result_actual.html', generated_text=generated_text,generated_song=generated_song, new_lyrics=markov_song,
                               verse_length=verse_length, chorus_length=chorus_length, bridge_length=bridge_length, rhyme_scheme=rhyme_scheme)
@app.route('/download-pdf')
def download_pdf():
    pdf_path = 'generated_song.pdf'  # Path to the compiled PDF file
    return send_file(pdf_path, as_attachment=True)

@app.route('/tune-generation', methods=['GET'])
def tune_generation_page():
    # Replace this with the logic to render the Tune_generation.html page
    return render_template('Tune_generation.html')   
@app.route('/MUSIC_TUNES', )
def hole():
    return render_template('MUSIC_TUNES.html')

@app.route('/GENERATE_TUNES', methods=['GET', 'POST'])
def GENERATE_TUNES():
    pdf_path = None
    if request.method == 'POST':
        Generated_Song = request.form['Generated_Song']
    import subprocess
    txt = Generated_Song
    pre = txt.split('\n')
    ans = []
    genre = ''
    artist = ''
    for i in pre:
            if(i.split(':')[0]=="Genre" or i.split(':')[0]=="Artist" or i.split(':')[0]=="Rhyme Scheme"):
                if(i.split(':')[0]=="Genre"):
                    genre = i.split(':')[1].strip()
                    
                elif(i.split(':')[0]=="Artist"):
                    artist = i.split(':')[1].strip()
                continue
                
            if(i.split('(')[0]==''):
                continue
            
            i+='\n'
            ans.append(i)

    txt = ''.join(ans)
    new_txt = ""

    for i in txt:
            if(i=="'" or i=='"' or i=="?" or i=="!"):
                continue
            new_txt += i

    char2notes = { 
    ' ':("a4 a4 ", "r2 "),
    'a':("<c a>2 ", "<e' a'>2 "),
    'b':("e2 ", "e'4 <e' g'> "),
    'c':("g2 ", "d'4 e' "),
    'd':("e2 ", "e'4 a' "),
    'e':("<c g>2 ", "a'4 <a' c'> "),
    'f':("a2 ", "<g' a'>4 c'' "),
    'g':("a2 ", "<g' a'>4 a' "),
    'h':("r4 g ", " r4 g' "),
    'i':("<c e>2 ", "d'4 g' "),
    'j':("a4 a ", "g'4 g' "),
    'k':("a2 ", "<g' a'>4 g' "),
    'l':("e4 g ", "a'4 a' "),
    'm':("c4 e ", "a'4 g' "),
    'n':("e4 c ", "a'4 g' "),
    'o':("<c a g>2  ", "a'2 "),
    'p':("a2 ", "e'4 <e' g'> "),
    'q':("a2 ", "a'4 a' "),
    'r':("g4 e ", "a'4 a' "),
    's':("a2 ", "g'4 a' "),
    't':("g2 ", "e'4 c' "),
    'u':("<c e g>2  ", "<a' g'>2"),
    'v':("e4 e ", "a'4 c' "),
    'w':("e4 a ", "a'4 c' "),
    'x':("r4 <c d> ", "g' a' "),
    'y':("<c g>2  ", "<a' g'>2"),
    'z':("<e a>2 ", "g'4 a' "),
    '\n':("r1 r1 ", "r1 r1 "),
    ',':("r2 ", "r2"),
    '.':("<c e a>2 ", "<a c' e'>2"),
    '5': ("<e g>2 ", "g'4 a' ")
    }

    upper_staff = ""
    lower_staff = ""
    for i in new_txt.lower():
        if i in char2notes:
            (l, u) = char2notes[i]
            upper_staff += u
            lower_staff += l

    staff = "{\n\\new PianoStaff << \n"
    staff += "  \\new Staff {" + upper_staff + "}\n"
    staff += "  \\new Staff { \clef bass " + lower_staff + "}\n"
    staff += ">>\n}\n"

    title = """\header {
        title = "Random"
        composer = "Random"
        tagline = "Copyright: Random"
        }"""

    new_title = []
    for i in title.split("\n"):
            if(len(i.split('"'))>1):
                if(i.split('"')[0]=='  title = '):
                    temp = i.split('"')[0]+'"'+genre+'"'
                    
                elif(i.split('"')[0]=='  composer = '):
                    temp = i.split('"')[0]+'"'+artist+'"'
                    
                else:
                    temp = i.split('"')[0]+'"'+'Me'+'"'

                new_title.append(temp)
            else:
                new_title.append(i)

    new_title = '\n'.join(new_title)

    lilypond_code = new_title + staff

        # LilyPond command to compile the 'simple.ly' file
                # Convert the LilyPond code to a PDF
    with open('generated_song.ly', 'w') as f:
            f.write(lilypond_code)

    path = r'"C:/Program Files (x86)/LilyPond/usr/bin/lilypond"'
    lilypond_command = f'{path} -fpdf generated_song.ly'

    try:
            subprocess.run(lilypond_command, shell=True, check=True)
            print("LilyPond compilation successful!")
            pdf_path = 'generated_song.pdf'
    except subprocess.CalledProcessError as e:
            print(f"LilyPond compilation failed with error: {e}")

    return render_template('MUSIC_TUNES.html', Generated_Song=Generated_Song, pdf_path=pdf_path)

@app.route("/Song_chatbot", methods=["GET", "POST"])
def Song_chatbot():
    generated_lyrics = ""
    if request.method == "POST":
        generated_song = request.args.get('generated_song')
        rhyme_scheme = request.form.get("rhyme_scheme")
        Artist = request.form.get("artist")
        Genre = request.form.get("genre")
        generated_song = request.form.get("generated_song")
        user_input = f"rhyme_scheme: {rhyme_scheme}\nArtist: {Artist}\nGenre: {Genre}\ngenerated_song: {generated_song}\n"
        generated_lyrics = generate_song_lyrics(user_input)

    return render_template("result_actual.html", generated_lyrics=generated_lyrics,generated_song=generated_song, rhyme_scheme=rhyme_scheme)

if __name__ == '__main__':
    app.run(debug=True)
