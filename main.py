import numpy as np
import pandas as pd
import flask
from flask import Flask, render_template, request
from urllib.parse import urlparse
from scrapper import get_comments
# from helper_functions import view_emotions
from lstm_loader import view_emotions_lstm

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def create_app():
    @app.route('/')
    def home():
        return render_template('home.html')

    '''def is_valid_url(url):
        parsed_url = urlparse(url)
        return parsed_url.scheme in ["http", "https"] and bool(parsed_url.netloc)'''

    @app.route('/results',methods=['POST'])
    def result():
        url = request.form.get('url')
        '''if not is_valid_url(url):
            error_msg = "Error: Invalid URL"
            return render_template('home.html', error_msg=error_msg)
        try:'''
        #url.raise_for_status()
        comments = get_comments(url)
        n, ang, love, fear, joy, sad, sur, e_no, comments, labels = view_emotions_lstm(comments)
        return render_template('result.html',n=n,ang=ang,love=love,fear=fear,joy=joy,sad=sad,sur=sur,e_no=e_no,comments=comments,labels=labels)
        '''except Exception as e:
            error_msg = f"Error: Invalid URL"
            return render_template('home.html',error_msg = error_msg)'''
    return app

if __name__ == '__main__':
    app = create_app()
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

