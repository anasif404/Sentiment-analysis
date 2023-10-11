import flask

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/greet', methods=['POST'])
def greet():
    values = request.form['name']
    return render_template('greet.html', name=values)


if __name__ == '__main__':
    app.run()